class Registry:
    """
    Generic registry to map string names/aliases to classes.
    """
    def __init__(self, name: str):
        self._name = name
        self._registry = {}

    def register(self, *aliases):
        """Decorator to register a class with multiple aliases."""
        def decorator(cls):
            # Use the first alias as the abbreviation, or fallback to class name
            cls.abbr = aliases[0].lower() if aliases else cls.__name__.lower()

            for alias in aliases:
                key = alias.lower()
                if key in self._registry:
                     print(f"Warning: Overwriting {self._name} registry key '{key}'")
                self._registry[key] = cls
            return cls
        return decorator

    def __getitem__(self, name: str):
        """Retrieve a class by name (case-insensitive)."""
        key = name.lower()
        try:
            return self._registry[key]
        except KeyError:
            available = self.list_available()
            error_msg = f"'{name}' not found in {self._name} registry. Available:\n"
            for cls_name, alias_list in available.items():
                error_msg += f"  {cls_name}: {', '.join(alias_list)}\n"
            raise ValueError(error_msg)

    def list_available(self):
        """Returns dict mapping class names to their aliases."""
        unique_classes = sorted(set(self._registry.values()), key=lambda x: x.__name__)
        return {
            cls.__name__: sorted([k for k, v in self._registry.items() if v == cls])
            for cls in unique_classes
        }
