# Computational, Programming, and Mathematical Methods

## SerialArm 

### Speeding Up fk and jacob

See sandbox/performance/fk_performance_evaluation.

### Weird Caching stuff

I'd like to be able to have functions cache, but it seems like setting up and lru cache makes the class unable to be pickled.
To get around this, I've made a potentially awkward arrangement where I have a set of "raw" functions, fk_raw and fk_atom_raw, that are used to create cached methods upon initialization.
These newly made cached functions can be deleted and their caches discarded when pickling, then setup again after unpickling.
This probably adds some overhead to instantiation but will pay off in the long run as I can do much smoother parallel processing stuff.

### Joint Limits

I'd like to be able to implement joint limits, but I don't want them to get in the way of stuff most of the time when the user doesn't want to consider them.
Ideally, you'd be able to include them and maybe choose between having functions like fk and jacob throw exceptions, warn but continue, or clamp.
I think I'll make the default joint limit be infinite and the default behavior be to clamp with warning.

### Inverse Kinematics

There's a challenge in making a single relatively simple ik function handle that can deal with many types of ik calls without needing excessive kwargs.
