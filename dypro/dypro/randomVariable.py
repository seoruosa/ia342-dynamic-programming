class ValueProbability():
    def __init__(self, value:tuple, probability:float):
            self.value = value
            self.probability = probability
            if(probability>1 or probability<0):
                raise TypeError
    
    def getValue(self):
        return self.value
    
    def getProbability(self):
        return self.probability
    
    def __str__(self):
        return f"{self.getProbability()} {self.getValue()}"

class RandomVariable(ValueProbability):
    def __init__(self, *args):
        if len(args)%2==0:
            # add valueProb to list
            pass
        else:
            raise Exception("Number of arguments is not pair")
        # super().__init__(value, probability)

if __name__ == "__main__":
    print('bla')
    RandomVariable(1,0.2, 2)