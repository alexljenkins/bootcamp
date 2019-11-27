# input1 = float(input("enter first input: "))
# input2 = float(input("enter second input: "))

def xor(input1,input2):
    #is input1 set but not input2?
    w1 = [1,-1]
    bias1 = 0
    dot1 = input1 * w1[0] + input2 * w1[1] + bias1
    out1 = 1.0 if dot1 > 0.5 else 0.0

    #is input2 set but not input1?
    w2 = [-1, 1]
    bias2 = 0
    dot2 = input1 * w2[0] + input2 * w2[1] + bias2
    out2 = 1.0 if dot2 > 0.5 else 0.0

    #did either out1 OR out2 fire?
    w3 = [1, 1]
    bias3 = 0
    dot3 = out1 * w3[0] + out2 * w3[1] + bias3
    out3 = 1.0 if dot3 > 0.5 else 0.0

    return out3

assert xor(0,0) == 0.0
assert xor(0,1) == 1.0
assert xor(1,0) == 1.0
assert xor(1,1) == 0.0
print(xor(0,0), xor(1,0), xor(0,1), xor(1,1))
