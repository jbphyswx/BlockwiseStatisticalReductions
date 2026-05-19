julia version of blockwise reductions


I think the easiest way is to support trivial N-ary, and rolling-window convolution reductions etc, and then support building trees of operations on top of them..

E.g. lowest level rolling window 100x100x100 and then after that binary reductions etc.. idk..


Importantly the plan should not duplicate work, should use its cache sparingly and ideally, etc.

We also need the streaming statistical reductions (like welfords algorithm, e.g. for single and blockwise reductions), handling of cache for counts and things..

For now let's support statistical moments and things like variances.

Then let's by default also support binning, so you could calculate e.g. rolling heatmaps.

Then also user defined functions, so long as they return a single output type over which all the operators we need are defined... (inferrable or provided I guess)


Maybe we should construct plan graph objects and then work through them idk...

We will also need support to write intermediates to disk (say we're working on too large data), we shouldnt need to store the entire plan (which could grow quadratic or exponentially) in memory, so writing intermeidates as we go should work.



TODO: Add NaN extension (do we use NaNStatistics? idk)
