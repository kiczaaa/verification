type Image = Tensor Rat [28, 28]
type Label = Index 10

validImage : Image -> Bool
validImage x = forall i j . 0 <= x ! i ! j <= 1

@network
classifier : Image -> Vector Rat 10

advises : Image -> Label -> Bool
advises x i = forall j . j != i => classifier x ! i > classifier x ! j

@parameter
epsilon : Rat

-- classification robustness & L_infit norm
boundedByEpsilon : Image -> Bool
boundedByEpsilon x = forall i j . -epsilon <= x ! i ! j <= epsilon

robustAround : Image -> Label -> Bool
robustAround image label = forall pertubation .
  let perturbedImage = image - pertubation in
  boundedByEpsilon pertubation and validImage perturbedImage =>
    advises perturbedImage label

@parameter(infer=True)
n : Nat

@dataset
trainingImages : Vector Image n

@dataset
trainingLabels : Vector Label n

@property
robust : Vector Bool n
robust = foreach i . robustAround (trainingImages ! i) (trainingLabels ! i)

--------------------------------------------------------------
-- translation to constraint loss functions (using dl2) 
-- for each samples, the `robustAround` function should be satisfied.
-- so we give penaly when 
-- boundedByEpsilon pertubaion 
-- and validImage perturbedImage
-- and not (advises perturbuedImage label) 
-- 
