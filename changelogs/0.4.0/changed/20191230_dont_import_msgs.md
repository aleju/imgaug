# Removed Outdated "Don't Import from this Module" Messages #539

The docstring of each module in ``imgaug.augmenters`` previously included a
suggestion to not directly import from that module, but instead use
``imgaug.augmenters.<AugmenterName>``. That was due to the categorization
still being unstable. As the categorization has now been fairly stable
for a long time, the suggestion was removed from all modules. Calling
``imgaug.augmenters.<AugmenterName>`` instead of
``imgaug.augmenters.<ModuleName>.<AugmenterName>`` is however still the
preferred way.
