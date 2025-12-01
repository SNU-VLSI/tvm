
class ImcflowFuncInOutOrderSetup:
  """
  setup order of input and output nodes of imcflow functions.
  IMCE has specific order of inputs. For example, 
  - input data : input arguments of imcflow function
    - some of them should be interleaved send.
  
  - output data : return values of imcflow function
    - some of them should be interleaved receive.

  - constant with CMD 
    - conv weight
      - weight : pushed by cmd. order is not important

  - constant
    - conv configuration : it should has priorify over conv input
    - minmax params
    - batch norm params
    - if a IMCE have multiple nodes which has constant node, receive order of the IMCE is from first op to last op in topological order. 

  order format:
    - 2D list format. outer list is order groups. inner list is interleaving group.
      e.g., [[input1, input2], [input3]] means input1 and input2 are interleaved send first, then input3 is sent.
    
  """