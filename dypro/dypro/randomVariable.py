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
        return f"{self.getValue()}:{self.getProbability()}"

class RandomVariable():
    def __init__(self, values:list, probabilities:list):
        if len(values) != len(probabilities):
            raise Exception("Number of values is not equal probabilities")
        elif sum(probabilities)!=1:
            raise Exception("Sum of probabilities is not 1")
        else:
            self.__randomValue = [ValueProbability(a, b) for (a, b) in zip(values, probabilities)]
    def randomValue(self):
        return self.__randomValue
    
    def __str__(self):
        return f"{'{'}{', '.join([str(v) for v in self.randomValue()])}{'}'}"

if __name__ == "__main__":
    a = RandomVariable((1,2,3), (0.2,0.5,0.3))
    print(a)