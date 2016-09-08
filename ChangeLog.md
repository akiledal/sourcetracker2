# sourcetracker2 changelog

## 2.0.1-dev (changes since 2.0.1 go here)

 * A unified API for sourcetracking with gibbs sampling, including
   leave-one-out cross-validation, has been created and is accessible as
   sourcetracker.gibbs.
 * Basic plotting and comparison functionality has been added.

 * A candidate public API has been created for both normal sink/source
   prediction and leave-one-out (LOO) classification. These calls are 
   ``_gibbs`` and ``_gibbs_loo``.
 * The per-sink feature assignments are recorded for every run and written to
   the output directory. They are named as X.contingency.txt where X is the
   name of a sink.

## 2.0.1

  * Initial alpha release.
  * Re-implements the Gibbs sampler from [@danknights's SourceTracker.](https://github.com/danknights/sourcetracker).
  * [click](http://click.pocoo.org/)-based command line interface through the ``sourcetracker2`` command.
  * Supports parallel execution using the `--jobs` parameter.
