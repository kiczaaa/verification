--------------------------------------------------------------------------------
-- Inputs

-- define a new name for the type of inputs of the network.
type InputVector = Vector Rat 4

-- add meaningful names for the input indices.
sepalLength = 0   -- measured in centimetres
sepalWidth  = 1   -- measured in centimetres
petalLength = 2   -- measured in centimetres
petalWidth  = 3   -- measured in centimetres

--------------------------------------------------------------------------------
-- Outputs

-- define output format - a vector of 3 rationals, 
-- each representing the score for the 3 classes.

type OutputVector = Vector Rat 3

-- add meaningful names for the output indices.
setosa      = 0
versicolor  = 1
virginica   = 2

--------------------------------------------------------------------------------
-- Network

-- use the `network` annotation to declare the name and the type of the network
@network
iris : InputVector -> OutputVector

-- --------------------------------------------------------------------------------
-- Check input data validity 
-- Define normal input ranges (based on training data - min, max values)
normalSepalLength : InputVector -> Bool
normalSepalLength x = 4.3 <= x ! sepalLength <= 7.9

normalSepalWidth : InputVector -> Bool
normalSepalWidth x = 2.0 <= x ! sepalWidth <= 4.4

normalPetalLength : InputVector -> Bool
normalPetalLength x = 1.0 <= x ! petalLength <= 6.9

normalPetalWidth : InputVector -> Bool
normalPetalWidth x = 0.1 <= x ! petalWidth <= 2.5

validInput : InputVector -> Bool
validInput x = normalSepalLength x and normalSepalWidth x 
    and normalPetalLength x and normalPetalWidth x
    and x ! sepalLength > x ! sepalWidth
    and x ! petalLength > x ! petalWidth

--------------------------------------------------------------------------------
@parameter(infer=True)
n : Nat
@parameter
epsilon : Rat

@dataset
trainingX : Vector InputVector n
@dataset
trainingy : Vector OutputVector n

-- -- auxiliary functions
-- advisedLabel : OutputVector -> Index 3
-- advisedLabel y = if (y ! 0 >= y ! 1 and y ! 0 >= y ! 2) then 0 else 
--     (if y ! 1 >= y ! 2 then 1 else 2)

-- with fold
max : Vector Rat 2 -> Rat
max v = if v ! 0 >= v ! 1 then v ! 0 else v ! 1


advisedLabel : inputVector -> Index 3
advisedLabel : 


advises : InputVector -> OutputVector -> Bool
advises input output = forall i . 
    let label = advisedLabel output in
    i != label => iris input ! label > iris input ! i 
    
boundedByEpsilon : InputVector -> Bool
boundedByEpsilon x = forall i . -epsilon <= x ! i <= epsilon

robustAround : InputVector -> OutputVector -> Bool
robustAround input output = forall pertubation .
    let perturbedInput = input - pertubation in 
    boundedByEpsilon pertubation and validInput perturbedInput => 
    advises input output  

@property
robust : Vector Bool n
robust = foreach i . robustAround (trainingX ! i) (trainingy ! i)