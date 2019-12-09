import random

class Person:
    def __init__(self, name):
        self.name = name
        self.state = 'bachelor'
        self.gen = self.markov()

    def __repr__(self):
        return f"The person {self.name} is {self.state}"

    def get_next_state(self):
        return next(self.gen)


    def markov(self):
        while True:
            yield self.state
            i = random.randint(1,10)
            if i == 10:
                self.state = 'dead'
                yield self.state
                break
            if self.state == 'bachelor':
                self.state = 'married'
            elif self.state =='married':
                i = random.randint(1,5)
                if i == 5:
                    self.state = 'divorced'



alex = Person("Alex")

alex.get_next_state()

alex
