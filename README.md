# Torcharchy
A PyTorch interface for Neural Network layers that implement hierarchical dependencies in the output.

## Explained with an example
Let's say you are training a policy for a videogame, and in the game you can take the following actions:
- move left
- move right
- move up
- move down
- charge
- shoot
- dodge
- deflect
- raise shield

Most common implementation would be an `nn.Linear` layer with 9 outputs, a softmax activation and some sort of KL divergence loss on top. However, this ignores a key existing relationship among the outputs and introduces dependencies between outputs in a way that is not consequent of such relationship. For example, if the policy chooses `charge` and a negative loss is applied, that loss influences all other actions equally, which is not 100% fair. However, if we acknowledge the hierarchical structure in the action space, then it looks something like this:
- move
  - left
  - right
  - up
  - down
- attack
  - charge
  - shoot
- defend
  - dodge
  - deflect
  - raise shield

By representing the output in this fashion we are allowing the policy to pick an action in stages, instead of having to bet everything on a single leaf node. The policy might have clear that the action must be an attack, but not sure about which specific attack to use. When given a positive or negative reward, this learning signal will also propagate following this structure. The usefulness of this representation is even more obvious if the policy needs to choose an action of a huge list. When we have hundreds of actions to choose from.
