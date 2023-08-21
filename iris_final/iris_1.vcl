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
irisClassifier : InputVector -> OutputVector

--------------------------------------------------------------------------------
-- Check input data validity 
-- 1) Define normal input ranges (based on training data - min, max values)
normalSepalLength : InputVector -> Bool
normalSepalLength x = 4.3 <= x ! sepalLength <= 7.9

normalSepalWidth : InputVector -> Bool
normalSepalWidth x = 2.0 <= x ! sepalWidth <= 4.4

normalPetalLength : InputVector -> Bool
normalPetalLength x = 1.0 <= x ! petalLength <= 6.9

normalPetalWidth : InputVector -> Bool
normalPetalWidth x = 0.1 <= x ! petalWidth <= 2.5

-- 2) Define relationship between feautures (observations from scatter plot)
-- sepalLength always longer than sepalWidth
-- petalLength always longer than petalWidth
normalLengths : InputVector -> Bool
normalLengths x = x ! sepalLength > x ! sepalWidth
    and x ! petalLength > x ! petalWidth

-- 3) Define relationship between feautures (from correlation analysis)
-- petalLength and petalWidth have high correlation
-- predicted petalWidth = petalLength * 0.4132382936645491 -0.35666804105655303
-- -0.6491527433673527 <= error <= 0.5574664034649219
-- -0.5 <= petalLength * 0.4132382936645491 - 0.35666804105655303 - petalWidth <= 0.5
minErr = -0.6491527433673527
maxErr = 0.5574664034649219
slope = 0.4132382936645491
intercept = - 0.35666804105655303
normalCorr : InputVector -> Bool
normalCorr x = minErr <= x ! petalLength * slope + intercept - x ! petalWidth <= maxErr

-- Combile all
validInput : InputVector -> Bool
validInput x = normalSepalLength x and normalSepalWidth x 
    and normalPetalLength x and normalPetalWidth x
    and normalLengths x
    and normalCorr x

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-- Property 1 - from scatter plot
-- If (petal length < 2) and (petal width < 0.5), then it is setosa
smallPetal : InputVector -> Bool
smallPetal x =
    x ! petalLength < 2 and x ! petalWidth < 0.5

isSetosa : InputVector -> Bool
isSetosa x = 
    let scores = irisClassifier x in 
    forall d . d != setosa => scores ! setosa > scores ! d

@property
property1 : Bool
property1 = forall x . validInput x and smallPetal x =>
    isSetosa x

--------------------------------------------------------------------------------
-- Property 2 - from scatter plot
-- If (sepal length < 6) and (petal length < 2), then it is setosa
lengths : InputVector -> Bool
lengths x =
    x ! sepalWidth > 3 and x ! petalLength < 2

@property
property2 : Bool
property2 = forall x . validInput x and lengths x =>
    isSetosa x

--------------------------------------------------------------------------------
-- Property 3 - Robustness
@parameter(infer=True)
n : Nat
@parameter
epsilon : Rat

@dataset
trainingX : Vector InputVector n
@dataset
trainingY : Vector OutputVector n

-- auxiliary functions
advisedLabel : OutputVector -> Index 3
advisedLabel y = if (y ! 0 >= y ! 1 and y ! 0 >= y ! 2) then 0 else 
    (if y ! 1 >= y ! 2 then 1 else 2)

advises : InputVector -> OutputVector -> Bool
advises input output = forall i . 
    let label = advisedLabel output in
    i != label => irisClassifier input ! label > irisClassifier input ! i 
    
boundedByEpsilon : InputVector -> Bool
boundedByEpsilon x = forall i . -epsilon <= x ! i <= epsilon

robustAround : InputVector -> OutputVector -> Bool
robustAround input output = forall pertubation .
    let perturbedInput = input + pertubation in 
    boundedByEpsilon pertubation and validInput perturbedInput => 
    advises perturbedInput output  

@property
robust : Vector Bool n
robust = foreach i . robustAround (trainingX ! i) (trainingY ! i)