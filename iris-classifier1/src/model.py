from __future__ import annotations
import collections
import csv
import datetime
import enum
from math import isclose
from pathlib import path
from typing import (
    cast,
    Any,
    Optional,
    overload,
    dataclass,
    Union,
    Iterator,
    Iterable,
    Callable,
    Protocol,
)
import weakref

from model import Sample, SampleDict

@dataclass
class Sample:
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    
@dataclass
class KnownSample(Sample):
    species: str

@dataclass
class TrainingKnownSample(KnownSample): 
    pass

@dataclass
class Hyperparameter:
    """k 및 거리 계산 알고리듬이 있는 튜닝 매개변수 집합"""
    
    k: int
    algorithm: Distance
    data: weakref.ReferenceType["TrainingData"]
    
    def classify(self, sample: Sample) -> str:
        """K-NN 알고리듬"""
        
@dataclass(frozen=True)
class Sample:
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    
@dataclass(frozen=True)
class KnownSample(Sample):
    species: str
    
@dataclass
class TestingKnownSample:
    sample: KnownSample
    classification: Optional[str] = None
    
@dataclass(frozen=True)
class TrainingKnownSample:
    """분류할 수 없음."""
    sample: KnownSample
    

class Sample:
    """Abstract superclass for all samples."""
    
    def __init__(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
        species: Optional[str] = None,
        ) -> None:
        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_length = petal_length
        self.petal_width = petal_width
        self.species = species
        self.classification: Optional[str] = None
        
    def __repr__(self) -> str:
        if self.species is None:
            known_unknown = "UnknownSample"
        else:
            known_unknown = "KnownSample"
        if self.classification is None:
            classification = ""
        else:
            classification = f",{self.classification}"
            
        return (
            f"{known_unknown}("
            f"sepal_length={self.sepal_length},"
            f"sepal_width={self.sepal_width},"
            f"petal_length={self.petal_length},"
            f"petal_width={self.petal_width},"
            f"species={self.species!r}"
            f"{classification}"
            f")"
        )
        
    def classify(self, classification: str) -> None:
        self.classification = classification
        
    def matches(self) -> bool:
        return self.species == self.classification
    
    
class Hyperparameter:
    """하이퍼파라미터 값과 전체 품질"""
    
    def __init__(self, k: int, training: "TrainingData") -> None:
        self.k = k
        self.data: weakref.ReferenceType["TrainingData"] = weakref.ref(training)
        self.quality: float
        
    def test(self) -> None:
        """잔체 테스트 스위트 실행"""
        training_data: Optional["TrainingData"] = self.data()
        if not training_data:
            raise RuntimeError("Broken Weak Reference")
        pass_count, fail_count = 0, 0
        for sample in training_data.testing:
            sample.classification = self.classify(sample)
            if sample.matches():
                pass_count += 1
            else:
                fail_count += 1
        self.quality = pass_count / (pass_count + fail_count)
        
        
class TrainingData:
    """샘플을 로드하고 테스트하는 메서드를 가지며, 학습 및 테스트 데이터셋을 포함한다."""
    
    def __init__(self, name: str) -> None:
        self.name = name
        self.uploaded: datetime.datetime
        self.tested: datetime.datetime
        self.training: list[Sample] = []
        self.testing: list[Sample] = []
        self.tuning: list[Hyperparameter] = []
          
    def load(self, raw_data_source: Iterable[dict[str, str]]) -> None:
        """원시 데이터 로드 및 분할"""
        for n, row in enumerate(raw_data_source):
            purpose = Purpose.Testing if n % 5 == 0 else Purpose.Training
            sample = KnownSample(
                sepal_length=float(row["sepal_length"]),
                sepal_width=float(row["sepal_width"]),
                petal_length=float(row["petal_length"]),
                petal_width=float(row["petal_width"]),
                species=row["species"],
            )
            if sample.purpose == Purpose.Testing:
                self.testing.append(sample)
            else:
                self.training.append(sample)
        self.uploaded = datetime.datetime.now(tz=datetime.datetime)
        
        {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
            "species": "Iris-setosa"
        }
        
    def test(self, parameter: Hyperparameter) -> None:
        """이 하이퍼파라미터 값으로 테스트한다."""
        parameter.test()
        self.tuning.append(parameter)
        self.tested = datetime.datetime.now(tz=datetime.timezone.utc)
        
    def classify(self, parameter: Hyperparameter, sample: Sample) -> Sample:
        """샘플을 분류한다."""
        Classification = parameter.classify(sample)
        sample.classify(Classification)
        return sample
    
    
class Distance:
    def distance(self, s1: Sample, s2:Sample) -> float:
        pass
    
    
class ED(Distance):
    def distance(self, s1: Sample, s2: Sample) -> float:
        return (
            s1.sepal_length - s2.sepal_length,
            s1.sepal_width - s2.sepal_width,
            s1.petal_length - s2.petal_length,
            s2.petal_width - s2.petal_width,
            )
        

class MD(Distance):
    def distance(self, s1: Sample, s2: Sample) -> float:
        return sum(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s2.petal_length),
                abs(s1.petal_width - s2.petal_width),               
            ]
        )
        
        
class CD(Distance):
    def distance(self, s1: Sample, s2: Sample) -> float:
        return max(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s2.petal_length),
                abs(s1.petal_width - s2.petal_width),
            ]
        )
    
    
class SD(Distance):
    def distance(self, s1: Sample, s2: Sample) -> float:
        return sum(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s2.petal_length),
                abs(s1.petal_width - s2.petal_width),
            ]
         ) / sum(
             [
                 s1.sepal_length + s2.sepal_length,
                 s1.sepal_width + s2.sepal_width,
                 s1.petal_length + s2.petal_length,
                 s1.petal_width + s2.petal_width,
             ]
         )
         
         
class TrainingKnownSample(KnownSample):
    @classmethod
    def from_dict(cls, row: dict[str, str]) -> "TrainingKnownSample":
        return cast(TrainingKnownSample, super.from_dict(row))
    

class KnownSample(Sample):
    
    def __init__(
        self, 
        species: str,
        sepal_length: float, 
        sepal_width: float, 
        petal_length: float, 
        petal_width: float, 
    ) -> None:
        super().__init__(
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width,
        )
        self.species = species
        
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"sepal_length={self.sepal_length},"
            f"sepal_width={self.sepal_length},"
            f"petal_length={self.petal_length},"
            f"petal_width={self.petal_width},"
            f"species={self.species!r},"
            f")"
            )
        
                
class TrainingData:
    
    def __init__(self, name: str) -> None:
        self.name = name
        self.uploaded: datetime.datetime
        self.tested: datetime.datetime
        self.training: list[KnownSample] = []
        self.testing: list[KnownSample] = []
        self.tuning: list[Hyperparameter] = []
        
    def load(self, raw_data_iter: Iterable[dict[str, str]]) -> None:
        for n, row in enumerate(raw_data_iter):
            try:
                if n % 5 == 0:
                    test = TestingKnownSample.from_dict(row)
                    self.testing.append(test)
                else:
                    train = TrainingKnownSample.from_dict(row)
                    self.training.append(train)
            except InvalidSampleError as ex:
                print(f"Row {n+1}: {ex}")
                return
        self.uploaded = datetime.datetime.now(tz=datetime.timezone.utc)
        

class SampleReader:
    
    target_class = Sample
    header = [
        "sepal_length", "sepal_width",
        "petal_length", "petal_width", "class"
    ]
    
    def __init__(self, source: path) -> None:
        self.source = source
        
    def Sample_iter(self) -> Iterator[Sample]:
        target_class = self.target_class
        with self.source.open() as source_file:
            reader = csv.DictReader(source_file, self.header)
            for row in reader:
                try:
                    sample = target_class(
                        sepal_length=float(row["sepal_length"]),
                        sepal_width=float(row["sepal_width"]),
                        petal_length=float(row["petal_length"]),
                        petal_width=float(row["petal_width"]),
                    )
                except ValueError as ex:
                    raise BadSampleRow(f"Invalid {row!r}") from ex
                yield sample
                

class BadSampleRow(ValueError):
    pass


class Purpose(enum.IntEnum):
    Classification = 0
    Testing = 1
    Training = 2
    
    
class KnownSample(Sample):
    
    def __init__(
        self, 
        sepal_length: float, 
        sepal_width: float, 
        petal_length: float,
        petal_width: float, 
        purpose: int,
        species: str
    ) -> None:
        Purpose_enum = Purpose(purpose)
        if Purpose_enum not in {Purpose.Training, Purpose.Testing}:
            raise ValueError(
                f"Invalid purpose: {purpose!r}: {Purpose_enum}"
            )
        super().__init__(
            sepal_length=sepal_length, 
            sepal_width=sepal_width, 
            petal_length=petal_length,
            petal_width=petal_width, 
           )
        self.purpose = Purpose_enum
        self.species = species
        self._Classification: Optional[str] = None
        
    def matches(self) -> bool:
        return self.species == self.classification
    
    @property
    def classification(self) -> Optional[str]:
        if self.purpose == Purpose.Testing:
            return self._Classification
        else:
            raise AttributeError(f"Training sample have no classification")
        
    @classification.setter
    def classification(self, value: str) -> None:
        if self.purpose == Purpose.Testing:
            self._classification - value
        else:
            raise AttributeError(
                 f"Training sample cannot be classified"
                
            )
            
            
class SamplePartition(list[SampleDict], abc.ABC ):

       
    def __init__(
        self,
        iterable: Optional[Iterable[SampleDict]] = None,
        *,
        training_subset: float = 0.80,
    ) -> None:
        self.training_subset = training_subset
        if iterable:
            super().__init__(iterable)
        else:
            super().__init__()
            
        @abc.abstractproperty
        @property
        def training(self) -> list[TrainingKnownSample]:
            
        @abc.abstractproperty
        @property
        def testing(self) -> list[TestingKnownSample]:
            
      
class ShufflingSamplePartition(SamplePartition):
    def __init__(
        self, 
        iterable: Optional[Iterable[SampleDict]] = None,
        *, 
        training_subset: float = 0.8,
        ) -> None:
        super().__init__(iterable, training_subset=training_subset)
        self.split: Optional[int] = None
        
        def shuffle(self) -> None:
            if not self.split:
                random.shuffle(self)
                self.split = int(len(self) * self.training_subset)
                
    @property
    def training(self) -> list[TrainingKnownSample]:
        self.shuffle()
        return[TrainingKnownSample(**sd) for sd in self[:self.split]]
    
    @property
    def testing(self) -> list[TestingKnownSample]:
        self.shuffle()
        return[TestingKnownSample(**sd) for sd in self[self.split :]]
    

class SampleDict(TypeDict):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    species: str
    
    
class shufflingPartition(SamplePartition):
    def __init__(
        self, 
        iterable: Optional[Iterable[SampleDict]] = None, 
        *, 
        training_subset: float = 0.8
    ) -> None:
        super().__init__(iterable, training_subset=training_subset)
        self.split: Optional[int] =None
        
    def shuffle(self) -> None:
        if not self.split:
            random.shuffle(self)
            self.split = int(len(self) *self.training_subset)
            
        @property
        def training(self) -> list[TrainingKnownSample]:
            self.shuffle()
            return [TrainingKnownSample(**sd) for sd in self[:self.split]]
        
        @property
        def testing(self) -> list[TesitngKnownSample]:
            self.shuffling()
            return [TestingKnownSample(**sd) for sd in self[self.split :]]
        
        ssp= ShufflingSamplePartition(training_subset=0.67)
        for row in data:
            ssp.append(row)
            

class DealingPartition(abc,ABC):
    @abc.abstractmethod
    def __init__(
        self,
        items: Optional[Iterable[SampleDict]],
        *,
        training_subset: tuple[int, int] = (8,10)
    ) -> None:
        
    @abc.abstractmethod
    def extend(self, items: Iterable[SampleDict]) -> None:
        
    @abc.abstractmethod
    def append(self, item: SampleDict) -> None:
        
    @abc.abstractmethod
    def testing(self) -> list[TestingKnownSample]:
        
          
class CountingDealingPartition(DealingPartition):
    def __init__(
        self, 
        items: Any | None, 
        *, 
        training_subset: tuple[int, int] = (8, 10),
        ) -> None:
        self.training_subset = training_subset
        self.counter = 0
        self._training: list[TrainingKnownSample] = []
        self._testing: list[TestingKnownSample] = []
        if items:
            self.extend(items)
            
    def extend(self, items: Iterable[SampleDict]) -> None:
        for item in items:
            self.append(item)
            
    def append(self, item: SampleDict) -> None:
        n, d = self.training_subset
        if self.counter % d < n:
            self._training.append(TrainingKnownSample(**item))
        else:
            self._testing.append(TestingKnownSample(**item))
            self.counter +=1
    @property
    def training(self) -> list[TrainingKnownSample]:
        return self._training
    
    @property
    def testing(self) -> list[TestingKnownSample]:
        return self._testing


class TestingKnownSample:
    def __init__(
        self, sample: KnownSample, Classification: Optional[str] = None
        ) -> None:
        self.sample = sample
        self.classification = Classification

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(sample={self.sample!r}),"
            f"classification={self.classification!r}"
        )
        

class TrainingKnownSample(NamedTuple):
    sample: KnownSample
    
__test__ = {name: case for name, case in globals().items() if name.startswith("test")}