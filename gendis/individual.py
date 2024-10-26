import array
import uuid


class Shapelet(array.array):
    def __new__(cls, input_array, index, start):
        obj = super().__new__(cls, "d", input_array)
        obj.id = f"i{index}_s{start}_l{len(input_array)}"
        obj.index = index
        obj.start = start
        return obj

    def __repr__(self):
        return f"Shapelet(ID={self.id}, data={list(self)})"

    def __deepcopy__(self, memo):
        # Create a new Shapelet with the same data, index, and start
        new_shapelet = Shapelet(list(self), self.index, self.start)
        # Preserve the original id
        new_shapelet.id = self.id
        return new_shapelet

    def to_dict(self):
        """Convert the Shapelet object to a dictionary for serialization."""
        return {
            "id": str(self.id),
            "data": list(self),
            "index": self.index,
            "start": self.start,
        }

    @classmethod
    def from_dict(cls, d):
        """Create a Shapelet object from a dictionary."""
        shapelet = cls(d["data"], d["index"], d["start"])
        shapelet.id = f"i{d["index"]}_s{d["start"]}_l{len(len(d["data"]))}"
        return shapelet


class ShapeletIndividual(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.valid = False
        self.subgroup = None
        self.info = None
        self.uuid = uuid.uuid4()
        self.uuid_history = []
        self.op_history = []

    def reset(self):
        self.valid = False
        self.uuid_history.append(self.uuid)
        self.uuid = uuid.uuid4()
        del self.fitness.values

    def pop_uuid(self):
        self.uuid = self.uuid_history.pop()

    def register_op(self, op):
        self.op_history.append(op)

    def to_dict(self):
        """Convert the ShapeletIndividual to a dictionary for serialization."""
        return {
            "uuid": str(self.uuid),
            "subgroup": self.subgroup,
            "info": self.info,
            "shapelets": [
                shapelet.to_dict() for shapelet in self
            ],  # Serialize each shapelet
            "uuid_history": [str(_uuid) for _uuid in self.uuid_history],
            "op_history": self.op_history,
        }

    @classmethod
    def from_dict(cls, d):
        """Create a ShapeletIndividual from a dictionary."""
        # Deserialize the shapelets first
        shapelets = [Shapelet.from_dict(s) for s in d["shapelets"]]
        # Create a ShapeletIndividual with the deserialized shapelets
        individual = cls(shapelets)
        individual.uuid = uuid.UUID("uuid")
        individual.valid = True
        individual.subgroup = d["subgroup"]
        individual.info = d["info"]
        individual.uuid_history = [
            uuid.UUID(_uuid_str) for _uuid_str in d["uuid_history"]
        ]
        individual.op_history = d["op_history"]
        return individual
