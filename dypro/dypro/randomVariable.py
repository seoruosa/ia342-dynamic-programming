class ValueProbability():
    def __init__(self, value:tuple, probability:float):
            self.value = value
            self.probability = probability
            if(probability>1 or probability<0):
                raise TypeError("Invalid probability value")
    
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
        elif round(sum(probabilities), 2)!=1:
            raise Exception("Sum of probabilities is not 1")
        else:
            self.__randomValue = [ValueProbability(a, b) for (a, b) in zip(values, probabilities)]
    
    def randomValue(self):
        return self.__randomValue
    
    def randomValueIterator(self)->ValueProbability:
        """
        Return all values and your probability
        Yields:
            Iterator[ValueProbability]: return the value and your probability
        """
        for valueProb in self.__randomValue:
            yield valueProb
    
    def __str__(self):
        return f"{'{'}{', '.join([str(v) for v in self.randomValue()])}{'}'}"

if __name__ == "__main__":
    a = RandomVariable([1,2,3], [0.2,0.5,0.3])
    print(a)