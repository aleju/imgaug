# Fixed `SnowflakesLayer` crash #471

* Fixed a crash in `SnowflakesLayer` that could occur when using values
  close to `1.0` for `flake_size`.
