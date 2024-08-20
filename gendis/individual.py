import uuid

class ShapeletIndividual(list):
    def __init__(self, *args):
        super().__init__(*args)
        # self.in_top_k = False
        self.valid = False
        self.subgroup = None
        self.info = None
        self.uuid = uuid.uuid4()
        self.uuid_history = []

    def reset(self):
        self.valid = False
        self.uuid_history.append(self.uuid)
        self.uuid = uuid.uuid4()
        del self.fitness.values

    def pop_uuid(self):
        if len(self.uuid_history) > 0:
            self.uuid = self.uuid_history.pop()
        else:
            self.uuid = uuid.uuid4()