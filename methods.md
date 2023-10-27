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
While I liked the challenge of exploring multiple IK methods used in RoboPy, I think in robopy2 I'm going to focus on getting just one really good method for each target type that the user could supply, at least for now.
IK types are therefore determined by the ways the user can represent target poses.
The cases to be considered are as follows:
1. XY
2. XYZ
3. XY-theta
4. Full Pose
5. Partially-unconstrained solving (e.g., solve for y but leave x and theta free, solve for x axis but leave y/z axes free)

Hmm, maybe I can write just one method that works for number (5) and have it apply to all other cases.
Then the user could also enter a target-mask, like a list [False False True False False True] that would indicate they wanted to solve for only z and z-axis pose, and any data-type could be given as a target

Just like with fk and jacob, there will be an ik and a _ik function, where ik parses and standardizes the user arguments to figure out what is really being solved for.
Then _ik can be cached (just like _fk is cached) and deal with a single argument pattern.

I could do a weighted pseudo-inverse and weight all masked DOF's to zero.
But that might be very inefficient for highly masked cases (like 2D movement).

## Visualization

### Architecture

