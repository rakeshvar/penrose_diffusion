from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import torch

from code.utils.advanced import sample_ot_noise, xya_to_scaled
from code.utils.lossy import ScipyBatchedLSA, gather_by_permutation, ot_cost_matrix


@dataclass
class PreparedOTBatch:
    x0: torch.Tensor
    noise: torch.Tensor
    colors: torch.Tensor
    labels: torch.Tensor


@dataclass
class _BufferSlot:
    in_use: bool = False
    cost_cpu: torch.Tensor | None = None
    colors_cpu: torch.Tensor | None = None
    permutation_cpu: torch.Tensor | None = None
    release_event: torch.cuda.Event | None = None


@dataclass
class _PendingOTBatch:
    slot: _BufferSlot | None
    x0: torch.Tensor
    noise: torch.Tensor
    colors: torch.Tensor
    labels: torch.Tensor
    ready_event: torch.cuda.Event | None = None
    future: Future | None = None
    prepared: PreparedOTBatch | None = None


class OTBatchPrefetcher:
    """Prepare and match batch k+1 while batch k trains."""

    def __init__(
        self,
        device,
        augmenter,
        max_workers=None,
        seed=None,
        async_enabled=True,
    ):
        self.device = torch.device(device)
        self.augmenter = augmenter
        self.solver = ScipyBatchedLSA(max_workers=max_workers)
        self.async_enabled = bool(async_enabled and self.device.type == "cuda")
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(torch.initial_seed() if seed is None else seed)
        self.wait_times_ms = []
        self._closed = False
        self._next_slot = 0
        self._slots = [_BufferSlot(), _BufferSlot()]

        if self.async_enabled:
            self._stream = torch.cuda.Stream(device=self.device)
            self._coordinator = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="ot-lsa-coordinator",
            )
        else:
            self._stream = None
            self._coordinator = None

    def _allocate_slot(self, slot, batch, tiles, dtype, colors_dtype):
        cost_shape = (batch, tiles, tiles)
        colors_shape = (batch, tiles)
        permutation_shape = (batch, tiles)
        if slot.cost_cpu is None or slot.cost_cpu.shape != cost_shape or slot.cost_cpu.dtype != dtype:
            slot.cost_cpu = torch.empty(
                cost_shape,
                dtype=dtype,
                device="cpu",
                pin_memory=True,
            )
        if (
            slot.colors_cpu is None
            or slot.colors_cpu.shape != colors_shape
            or slot.colors_cpu.dtype != colors_dtype
        ):
            slot.colors_cpu = torch.empty(
                colors_shape,
                dtype=colors_dtype,
                device="cpu",
                pin_memory=True,
            )
        if slot.permutation_cpu is None or slot.permutation_cpu.shape != permutation_shape:
            slot.permutation_cpu = torch.empty(
                permutation_shape,
                dtype=torch.long,
                device="cpu",
                pin_memory=True,
            )
        if slot.release_event is None:
            slot.release_event = torch.cuda.Event()

    def _acquire_slot(self, batch, tiles, dtype, colors_dtype):
        for offset in range(len(self._slots)):
            index = (self._next_slot + offset) % len(self._slots)
            slot = self._slots[index]
            if slot.in_use:
                continue
            if slot.release_event is not None:
                slot.release_event.synchronize()
            self._allocate_slot(slot, batch, tiles, dtype, colors_dtype)
            slot.in_use = True
            self._next_slot = (index + 1) % len(self._slots)
            return slot
        raise RuntimeError("Both OTFM prefetch buffers are still in use")

    def _solve_after_copy(self, event, slot):
        event.synchronize()
        assert slot.cost_cpu is not None
        assert slot.colors_cpu is not None
        assert slot.permutation_cpu is not None
        permutation = self.solver.solve_numpy(
            slot.cost_cpu.numpy(),
            slot.colors_cpu.numpy(),
        )
        slot.permutation_cpu.copy_(torch.from_numpy(permutation))

    def _prepare_async(self, xya, colors, labels):
        batch, tiles, _ = xya.shape
        slot = self._acquire_slot(batch, tiles, xya.dtype, colors.dtype)
        assert self._stream is not None
        assert self._coordinator is not None
        assert slot.cost_cpu is not None
        assert slot.colors_cpu is not None

        with torch.cuda.stream(self._stream):
            xya_device = xya.to(self.device, non_blocking=True)
            colors_device = colors.to(self.device, non_blocking=True)
            labels_device = labels.to(self.device, non_blocking=True)
            augmented = self.augmenter(xya_device, generator=self.generator)
            x0, _ = xya_to_scaled(augmented)
            noise = sample_ot_noise(
                x0.shape,
                device=self.device,
                dtype=x0.dtype,
                generator=self.generator,
            )
            cost = ot_cost_matrix(x0, noise)
            slot.cost_cpu.copy_(cost, non_blocking=True)
            slot.colors_cpu.copy_(colors_device, non_blocking=True)
            ready_event = torch.cuda.Event()
            ready_event.record(self._stream)

        future = self._coordinator.submit(self._solve_after_copy, ready_event, slot)
        return _PendingOTBatch(
            slot=slot,
            x0=x0,
            noise=noise,
            colors=colors_device,
            labels=labels_device,
            ready_event=ready_event,
            future=future,
        )

    def _prepare_synchronous(self, xya, colors, labels):
        xya_device = xya.to(self.device)
        colors_device = colors.to(self.device)
        labels_device = labels.to(self.device)
        augmented = self.augmenter(xya_device, generator=self.generator)
        x0, _ = xya_to_scaled(augmented)
        noise = sample_ot_noise(
            x0.shape,
            device=self.device,
            dtype=x0.dtype,
            generator=self.generator,
        )
        permutation = self.solver.solve(ot_cost_matrix(x0, noise), colors_device)
        prepared = PreparedOTBatch(
            x0=x0,
            noise=gather_by_permutation(noise, permutation),
            colors=colors_device,
            labels=labels_device,
        )
        return _PendingOTBatch(
            slot=None,
            x0=x0,
            noise=noise,
            colors=colors_device,
            labels=labels_device,
            prepared=prepared,
        )

    def prepare(self, batch):
        if self._closed:
            raise RuntimeError("OTBatchPrefetcher is closed")
        xya, colors, labels = batch
        if self.async_enabled:
            return self._prepare_async(xya, colors, labels)
        return self._prepare_synchronous(xya, colors, labels)

    def consume(self, pending):
        if pending.prepared is not None:
            return pending.prepared

        assert pending.future is not None
        assert pending.ready_event is not None
        assert pending.slot is not None
        assert pending.slot.permutation_cpu is not None

        started = time.perf_counter()
        pending.future.result()
        self.wait_times_ms.append((time.perf_counter() - started) * 1_000.)

        current_stream = torch.cuda.current_stream(self.device)
        current_stream.wait_event(pending.ready_event)
        permutation = pending.slot.permutation_cpu.to(self.device, non_blocking=True)
        matched_noise = gather_by_permutation(pending.noise, permutation)

        for tensor in (pending.x0, pending.noise, matched_noise, pending.colors, pending.labels):
            tensor.record_stream(current_stream)

        prepared = PreparedOTBatch(
            x0=pending.x0,
            noise=matched_noise,
            colors=pending.colors,
            labels=pending.labels,
        )
        assert pending.slot.release_event is not None
        pending.slot.release_event.record(current_stream)
        pending.slot.in_use = False
        return prepared

    def iter_prepared(self, loader):
        if not self.async_enabled:
            for batch in loader:
                yield self.consume(self.prepare(batch))
            return

        iterator = iter(loader)
        try:
            current = self.prepare(next(iterator))
        except StopIteration:
            return

        for batch in iterator:
            following = self.prepare(batch)
            yield self.consume(current)
            current = following
        yield self.consume(current)

    @property
    def mean_wait_ms(self):
        if not self.wait_times_ms:
            return 0.
        return sum(self.wait_times_ms) / len(self.wait_times_ms)

    def close(self):
        if self._closed:
            return
        if self._coordinator is not None:
            self._coordinator.shutdown(wait=True)
        self.solver.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
