import json
from datetime import date
from pathlib import Path
from typing import List, Optional, Union

from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon


class Sample:
    """
    Represents a water sample with all associated measurements and observations.
    """

    def __init__(
        self,
        sample_id: str,
        long: float,
        lat: float,
        water_body_type: str,
        date: Union[str, date],
        unnamed_0: Optional[int] = None,
        dynamic_risk_assessment: Optional[str] = None,
        estimated_width: Optional[float] = None,
        estimated_depth: Optional[float] = None,
        water_flow: Optional[str] = None,
        water_level: Optional[str] = None,
        temperature: Optional[float] = None,
        total_dissolved_solids: Optional[float] = None,
        turbidity: Optional[float] = None,
        phosphate: Optional[float] = None,
        ph: Optional[float] = None,
        nitrate: Optional[str] = None,
        ammonia: Optional[float] = None,
        pollution_evidence: Optional[List[str]] = None,
        pollution_source: Optional[List[str]] = None,
        flow_impedance: Optional[List[str]] = None,
        invasive_plant: Optional[List[str]] = None,
        wildlife: Optional[List[str]] = None,
        bank_vegetation: Optional[List[str]] = None,
        land_use: Optional[List[str]] = None,
    ):
        """
        Initialize a Sample object.

        Args:
            sample_id: Unique identifier for the sample
            long: Longitude coordinate
            lat: Latitude coordinate
            water_body_type: Type of water body (e.g., 'stream', 'river')
            date: Date of sample collection
            unnamed_0: Index value from original data
            dynamic_risk_assessment: Risk assessment value
            estimated_width: Estimated width of water body in meters
            estimated_depth: Estimated depth of water body in meters
            water_flow: Description of water flow
            water_level: Description of water level
            temperature: Water temperature in degrees Celsius
            total_dissolved_solids: TDS measurement
            turbidity: Turbidity measurement
            phosphate: Phosphate measurement
            ph: pH measurement
            nitrate: Nitrate measurement
            ammonia: Ammonia measurement
            pollution_evidence: List of pollution evidence observed
            pollution_source: List of pollution sources identified
            flow_impedance: List of flow impedance factors
            invasive_plant: List of invasive plants observed
            wildlife: List of wildlife observed
            bank_vegetation: List of bank vegetation types
            land_use: List of land use types in the area
        """
        self.unnamed_0 = unnamed_0
        self.id = sample_id
        self.long = long
        self.lat = lat
        self.water_body_type = water_body_type
        self.dynamic_risk_assessment = dynamic_risk_assessment
        self.estimated_width = estimated_width
        self.estimated_depth = estimated_depth
        self.water_flow = water_flow
        self.water_level = water_level
        self.temperature = temperature
        self.total_dissolved_solids = total_dissolved_solids
        self.turbidity = turbidity
        self.phosphate = phosphate
        self.ph = ph
        self.nitrate = nitrate
        self.ammonia = ammonia
        self.date = date if isinstance(date, str) else date.isoformat()
        self.pollution_evidence = pollution_evidence or []
        self.pollution_source = pollution_source or []
        self.flow_impedance = flow_impedance or []
        self.invasive_plant = invasive_plant or []
        self.wildlife = wildlife or []
        self.bank_vegetation = bank_vegetation or []
        self.land_use = land_use or []

    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> "Sample":
        """
        Read a Sample object from a JSON file.

        The file should contain a single point entry with a key (e.g., "6871")
        containing the sample data as a dictionary.

        Args:
            filepath: Path to the JSON file

        Returns:
            Sample object created from the file data

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, "r") as f:
            data = json.load(f)

        # The file contains a dictionary with one key (the point ID)
        # Extract the nested dictionary
        if not data:
            raise ValueError("File contains no data")

        # Get the first (and should be only) entry
        point_key = list(data.keys())[0]
        point_data = data[point_key]

        # Convert camelCase keys to snake_case for constructor
        return cls(
            unnamed_0=point_data.get("Unnamed: 0"),
            sample_id=point_data["id"],
            long=point_data["long"],
            lat=point_data["lat"],
            water_body_type=point_data["waterBodyType"],
            dynamic_risk_assessment=point_data.get("dynamicRiskAssessment"),
            estimated_width=point_data.get("estimatedWidth"),
            estimated_depth=point_data.get("estimatedDepth"),
            water_flow=point_data.get("waterFlow"),
            water_level=point_data.get("waterLevel"),
            temperature=point_data.get("temperature"),
            total_dissolved_solids=point_data.get("totalDissolvedSolids"),
            turbidity=point_data.get("turbidity"),
            phosphate=point_data.get("phosphate"),
            ph=point_data.get("ph"),
            nitrate=point_data.get("nitrate"),
            ammonia=point_data.get("ammonia"),
            date=point_data["Date"],
            pollution_evidence=point_data.get("pollutionEvidence", []),
            pollution_source=point_data.get("pollutionSource", []),
            flow_impedance=point_data.get("flowImpedance", []),
            invasive_plant=point_data.get("invasivePlant", []),
            wildlife=point_data.get("wildlife", []),
            bank_vegetation=point_data.get("bankVegetation", []),
            land_use=point_data.get("landUse", []),
        )

    def to_dict(self, point_key: Optional[str] = None) -> dict:
        """
        Convert the Sample object to a dictionary matching the original format.

        Args:
            point_key: Optional key to wrap the data (e.g., "6871").
                      If None, returns unwrapped dictionary.

        Returns:
            Dictionary representation of the sample
        """
        data = {
            "Unnamed: 0": self.unnamed_0,
            "id": self.id,
            "long": self.long,
            "lat": self.lat,
            "waterBodyType": self.water_body_type,
            "dynamicRiskAssessment": self.dynamic_risk_assessment,
            "estimatedWidth": self.estimated_width,
            "estimatedDepth": self.estimated_depth,
            "waterFlow": self.water_flow,
            "waterLevel": self.water_level,
            "temperature": self.temperature,
            "totalDissolvedSolids": self.total_dissolved_solids,
            "turbidity": self.turbidity,
            "phosphate": self.phosphate,
            "ph": self.ph,
            "nitrate": self.nitrate,
            "ammonia": self.ammonia,
            "Date": self.date,
            "pollutionEvidence": self.pollution_evidence,
            "pollutionSource": self.pollution_source,
            "flowImpedance": self.flow_impedance,
            "invasivePlant": self.invasive_plant,
            "wildlife": self.wildlife,
            "bankVegetation": self.bank_vegetation,
            "landUse": self.land_use,
        }

        if point_key is not None:
            return {point_key: data}
        return data

    def to_file(
        self,
        filepath: Union[str, Path],
        point_key: Optional[str] = None,
        indent: int = 4,
    ):
        """
        Write the Sample object to a JSON file.

        Args:
            filepath: Path where the file should be written
            point_key: Optional key to wrap the data (e.g., "6871")
            indent: Number of spaces for JSON indentation (default: 4)
        """
        filepath = Path(filepath)

        # Create parent directories if they don't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict(point_key=point_key)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=indent)

    def __repr__(self) -> str:
        """String representation of the Sample object."""
        return (
            f"Sample(id='{self.id}', "
            f"location=({self.lat}, {self.long}), "
            f"date='{self.date}', "
            f"water_body_type='{self.water_body_type}')"
        )

    def __str__(self) -> str:
        """User-friendly string representation."""
        return (
            f"Water Sample {self.id}\n"
            f"Location: ({self.lat}, {self.long})\n"
            f"Date: {self.date}\n"
            f"Type: {self.water_body_type}\n"
            f"Temperature: {self.temperature}°C\n"
            f"pH: {self.ph}"
        )


def get_catchment_form_point(point: Point) -> Polygon:
    raise NotImplementedError


def get_water_quality_samples_for_poly(area: Polygon) -> List[Sample]:
    raise NotImplementedError


def get_water_quality_samples_for_point(point: Point) -> List[Sample]:
    poly = get_catchment_form_point(point)
    return get_water_quality_samples_for_poly(poly)
