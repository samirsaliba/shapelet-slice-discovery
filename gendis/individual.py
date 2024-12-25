import array
import uuid
import random

# Set a seed for reproducibility
SEED = 633458965612378623578
rng = random.Random(SEED)  # Create a random generator with a fixed seed


class Shapelet(array.array):
    def __new__(cls, input_array, index, start):
        obj = super().__new__(cls, "d", input_array)
        obj.__id = f"i{index}_s{start}_l{len(input_array)}"
        obj.__index = index
        obj.__start = start
        obj.__history = []
        return obj

    def __repr__(self):
        return f"Shapelet(ID={self.id}, data={list(self)})"

    @property
    def id(self):
        return self.__id

    @property
    def index(self):
        return self.__index

    @property
    def start(self):
        return self.__start

    @property
    def history(self):
        return self.__history

    def register_op(self, op):
        self.__history.append(op)
        self.__id += op

    def __deepcopy__(self, memo):
        return Shapelet(list(self).copy(), index=self.__index, start=self.__start)

    def to_dict(self):
        """Convert the Shapelet object to a dictionary for serialization."""
        return {
            "id": str(self.__id),
            "data": list(self),
            "index": self.__index,
            "start": self.__start,
        }

    @classmethod
    def from_dict(cls, d):
        """Create a Shapelet object from a dictionary."""
        shapelet = cls(d["data"], d["index"], d["start"])
        return shapelet


class ShapeletIndividual(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.valid = False
        self.subgroup = None
        self.subgroup_size = 1
        self.info = None
        # Generate reproducible UUID based on the seeded random generator
        self.__uuid = uuid.UUID(int=rng.getrandbits(128))
        self.__uuid_history = []
        self.__op_history = []

    @property
    def uuid(self):
        return self.__uuid

    @property
    def uuid_history(self):
        return self.__uuid_history

    @property
    def op_history(self):
        return self.__op_history

    def __from_dict_set_properties(self, _uuid, uuid_history, op_history):
        self.__uuid = _uuid
        self.__uuid_history = uuid_history
        self.__op_history = op_history

    def reset(self):
        self.valid = False
        self.__uuid_history.append(self.__uuid)
        # Generate a new reproducible UUID each time reset is called
        self.__uuid = uuid.UUID(int=rng.getrandbits(128))

        del self.fitness.values

    def pop_uuid(self):
        self.__uuid = self.__uuid_history.pop()

    def register_op(self, op):
        self.__op_history.append(op)

    def to_dict(self):
        """Convert the ShapeletIndividual to a dictionary for serialization."""
        return {
            "__uuid": str(self.__uuid),
            "subgroup": self.subgroup,
            "info": self.info,
            "shapelets": [
                shapelet.to_dict() for shapelet in self
            ],  # Serialize each shapelet
            "__uuid_history": [str(_uuid) for _uuid in self.__uuid_history],
            "__op_history": self.__op_history,
        }

    @classmethod
    def from_dict(cls, d):
        """Create a ShapeletIndividual from a dictionary."""
        # Deserialize the shapelets first
        shapelets = [Shapelet.from_dict(s) for s in d["shapelets"]]
        # Create a ShapeletIndividual with the deserialized shapelets
        individual = cls(shapelets)
        individual.__from_dict_set_properties(
            _uuid=uuid.UUID(d["__uuid"]),
            uuid_history=[uuid.UUID(_uuid_str) for _uuid_str in d["__uuid_history"]],
            op_history=d["__op_history"],
        )
        individual.valid = True
        individual.subgroup = d["subgroup"]
        individual.info = d["info"]
        return individual
