import array
import copy
import uuid


class Shapelet(array.array):
    def __new__(cls, input_array):
        obj = super().__new__(cls, 'd', input_array)
        obj.id = str(uuid.uuid4())
        return obj

    def reset_id(self):
        """Resets the shapelet's ID to a new unique value."""
        self.id = str(uuid.uuid4())

    def __repr__(self):
        return f"Shapelet(ID={self.id}, data={list(self)})"


class ShapeletIndividual(list):
    def __init__(self, *args):
        super().__init__(*args)
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

    @staticmethod
    def clone(individual):
        """Custom clone function to ensure Shapelet objects are cloned correctly."""
        # Perform a deep copy to ensure all elements, including Shapelets, are cloned
        clone = copy.deepcopy(individual)

        for i, item in enumerate(clone):
            if isinstance(item, ShapeletIndividual):
                pass
            elif isinstance(item, array.array):
                clone[i] = Shapelet(item)
            else:
                raise TypeError(f"Wrong individual shapelet type, f{type(item)}") 

        return clone