from __future__ import annotations
import csv
from enum import Enum, auto
from functools import wraps
from pathlib import Path
from typing import (
    cast,
    Optional,
    Callable,
    overload,
    Union,
    Iterator,
    Iterable,
    Sequence,
    Mapping
)

from flask import Flask

app = Flask(__name__)

@app.route('user/<user_name>')
def get_user(user_name):
    return user_name

@app.route('/iris/<iris_name>')
def get_iris(iris_name):
    return iris_name


class actors:
    Botanist: "Botanist"
    Researcher: "Researcher"
    User: "User"
    

class User:
    
    def __init__(
        self,
        username: str,
        password: str,
        name: str,
        email: str,
        role: str,
        interest: str,     
        ) -> None:
        self.username = username
        self.password = password
        self.name = name
        self.email = email
        self.role = role
        self.interest = interest
        
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"username={self.username!r},"
            f"password={self.password!r},"
            f"name={self.name!r},"
            f"email={self.email!r},"
            f"interest={self.interest!r}"       
        )
    def _set_username(self, username: str) -> None:
        if not username:
            raise ValueError(f"Invalid username {username!r}")
        self._username = username
    
    def _set_password(self, password: str) -> None:
        if not password:
            raise ValueError(f"Invalid password {password!r}")
        self._password = password
           
    def _get_state(self) -> str:
        print(f"Getting {self._name}'s State")
        return 
   
    def __len__(self) -> int:
        return len(self.User_List)
    
         
class Iris_classifier:
    header = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
        
    def __init__(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
        species: str
        ) -> None:
        self.sepal_length = sepal_length,
        self.sepal_with = sepal_width,
        self.petal_length = petal_length,
        self.petal_width = petal_width,
        self.species = species
        
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"sepal_length={self.sepal_length!r},"
            f"sepal_width={self.sepal_with!r},"
            f"petal_length={self.petal_length!r},"
            f"petal_width={self.petal_width!r},"
            f"species={self.species!r},"
            f")"
            )
        
    def _set_state(self, state: str):
        print(f"Setting {self._name}'s State to {state!r}")
        self._state = state
        
    def search(self, name: str) -> list["Iris"]:
        matching_irises: list["Iris"] = []
        for iris in self:
            if name in iris.name:
                matching_irises.append(iris)
            return matching_irises
        

class InvalidSampleError(ValueError):
    """소스 데이터 파일이 유효하지 않은 데이터 표현을 가지고 있다."""
    
    @classmethod
    def from_dict(cls, row: dict[str,str]) -> KnownSample:
        if row["species"] not in {
            "Iris-setosa", "Iris-versicolour", "Iris-virginica"}:
            raise InvalidSampleError(f"invalid species in {row}")
        try:
            return cls(
                species=row["species"],
                sepal_length=float(row["sepal_length"]),
                sepal_width=float(row["sepal_width"]),
                petal_length=float(row["petal_length"]),
                petal_width=float(row["petal_width"]), 
            )
        except ValueError as ex:
            raise InvalidSampleError(f"invalid {row!r}")
        
if __name__ == "__main__":
    app.run(debug=True)