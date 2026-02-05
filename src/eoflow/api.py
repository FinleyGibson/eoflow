from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from shapely.geometry import Point

from eoflow.ea import EAWaterQualityAPI
from eoflow.water_samples import Sample, get_water_quality_samples_for_point

app = FastAPI(
    title="Water Quality API",
    description="API to retrieve water quality samples and EA monitoring data",
    version="1.0.0",
)


class PointRequest(BaseModel):
    """Request model for a geographic point"""

    longitude: float = Field(
        ..., description="Longitude coordinate", ge=-180, le=180
    )
    latitude: float = Field(
        ..., description="Latitude coordinate", ge=-90, le=90
    )

    class Config:
        json_schema_extra = {
            "example": {
                "longitude": -1.5,
                "latitude": 52.5,
            }
        }


class PolygonRequest(BaseModel):
    """Request model for a polygon area"""

    coordinates: List[List[float]] = Field(
        ...,
        description="List of [longitude, latitude] coordinate pairs defining the polygon",
        min_length=3
    )

    class Config:
        json_schema_extra = {
            "example": {
                "coordinates": [
                    [-4.5, 50.3],
                    [-4.5, 51.2],
                    [-3.0, 51.2],
                    [-3.0, 50.3],
                    [-4.5, 50.3]
                ]
            }
        }


class EAWaterQualityRequest(BaseModel):
    """Request model for EA water quality data"""

    polygon: PolygonRequest = Field(..., description="Polygon defining the area of interest")
    determinand: str = Field(..., description="Determinand code (e.g., '0076' for temperature)")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    area: str = Field(
        ...,
        description="Precanned area code (e.g., 'environment_agency,SWX')"
    )
    verbose: bool = Field(default=False, description="Whether to show verbose logging")

    class Config:
        json_schema_extra = {
            "example": {
                "polygon": {
                    "coordinates": [
                        [-4.5, 50.3],
                        [-4.5, 51.2],
                        [-3.0, 51.2],
                        [-3.0, 50.3],
                        [-4.5, 50.3]
                    ]
                },
                "determinand": "0076",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "area": "environment_agency,SWX",
                "verbose": False
            }
        }


class EAMultipleDeterminandsRequest(BaseModel):
    """Request model for multiple EA determinands"""

    polygon: PolygonRequest = Field(..., description="Polygon defining the area of interest")
    determinands: Dict[str, str] = Field(
        ...,
        description="Dictionary mapping determinand codes to column names"
    )
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    area: str = Field(
        ...,
        description="Precanned area code (e.g., 'environment_agency,SWX')"
    )
    verbose: bool = Field(default=False, description="Whether to show verbose logging")

    class Config:
        json_schema_extra = {
            "example": {
                "polygon": {
                    "coordinates": [
                        [-4.5, 50.3],
                        [-4.5, 51.2],
                        [-3.0, 51.2],
                        [-3.0, 50.3],
                        [-4.5, 50.3]
                    ]
                },
                "determinands": {
                    "0076": "Temperature",
                    "0077": "Conductivity"
                },
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "area": "environment_agency,SWX",
                "verbose": False
            }
        }


class SampleResponse(BaseModel):
    """Response model for a water quality sample"""

    id: str
    long: float
    lat: float
    water_body_type: str
    date: str
    unnamed_0: int | None = None
    dynamic_risk_assessment: str | None = None
    estimated_width: float | None = None
    estimated_depth: float | None = None
    water_flow: str | None = None
    water_level: str | None = None
    temperature: float | None = None
    total_dissolved_solids: float | None = None
    turbidity: float | None = None
    phosphate: float | None = None
    ph: float | None = None
    nitrate: str | None = None
    ammonia: float | None = None
    pollution_evidence: List[str] = []
    pollution_source: List[str] = []
    flow_impedance: List[str] = []
    invasive_plant: List[str] = []
    wildlife: List[str] = []
    bank_vegetation: List[str] = []
    land_use: List[str] = []

    class Config:
        json_schema_extra = {
            "example": {
                "id": "sample_001",
                "long": -1.5,
                "lat": 52.5,
                "water_body_type": "stream",
                "date": "2024-01-15",
                "temperature": 12.5,
                "ph": 7.2,
                "pollution_evidence": ["plastic"],
                "wildlife": ["ducks", "fish"],
            }
        }


def sample_to_response(sample: Sample) -> SampleResponse:
    """Convert a Sample object to a SampleResponse model"""
    return SampleResponse(
        id=sample.id,
        long=sample.long,
        lat=sample.lat,
        water_body_type=sample.water_body_type,
        date=sample.date,
        unnamed_0=sample.unnamed_0,
        dynamic_risk_assessment=sample.dynamic_risk_assessment,
        estimated_width=sample.estimated_width,
        estimated_depth=sample.estimated_depth,
        water_flow=sample.water_flow,
        water_level=sample.water_level,
        temperature=sample.temperature,
        total_dissolved_solids=sample.total_dissolved_solids,
        turbidity=sample.turbidity,
        phosphate=sample.phosphate,
        ph=sample.ph,
        nitrate=sample.nitrate,
        ammonia=sample.ammonia,
        pollution_evidence=sample.pollution_evidence,
        pollution_source=sample.pollution_source,
        flow_impedance=sample.flow_impedance,
        invasive_plant=sample.invasive_plant,
        wildlife=sample.wildlife,
        bank_vegetation=sample.bank_vegetation,
        land_use=sample.land_use,
    )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Water Quality Samples API",
        "version": "1.0.0",
        "endpoints": {
            "/samples": "POST - Get water quality samples for a point",
            "/ea/water-quality": "POST - Get EA water quality data within a polygon",
            "/ea/water-quality/multiple": "POST - Get multiple EA determinands within a polygon",
            "/docs": "GET - Interactive API documentation",
        },
    }


@app.post("/samples", response_model=List[SampleResponse])
async def get_samples(point: PointRequest):
    """
    Get water quality samples for a given geographic point.

    Args:
        point: A PointRequest containing longitude and latitude coordinates

    Returns:
        A list of water quality samples for the catchment area containing the point

    Raises:
        HTTPException: If there's an error processing the request
    """
    try:
        # Create a shapely Point from the request coordinates
        shapely_point = Point(point.longitude, point.latitude)

        # Get samples for the point
        samples = get_water_quality_samples_for_point(shapely_point)

        # Convert Sample objects to response models
        response_samples = [sample_to_response(sample) for sample in samples]

        return response_samples

    except NotImplementedError:
        raise HTTPException(
            status_code=501,
            detail="The water quality sampling functionality is not yet fully implemented. "
            "Please ensure get_catchment_form_point and get_water_quality_samples_for_poly are implemented.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your request: {str(e)}",
        )


@app.post("/ea/water-quality")
async def get_ea_water_quality(request: EAWaterQualityRequest):
    """
    Get EA water quality data for a specific determinand within a polygon area.

    This endpoint fetches data from the Environment Agency Water Quality API
    for a specified area and determinand, then filters the results to only
    include observations within the provided polygon.

    Args:
        request: EAWaterQualityRequest containing polygon, determinand, dates, and area

    Returns:
        A dictionary containing:
        - total_records: Total number of records before filtering
        - filtered_records: Number of records within the polygon
        - data: List of records as dictionaries

    Raises:
        HTTPException: If there's an error processing the request
    """
    try:
        # Convert polygon coordinates to list of tuples
        polygon_coords = [(coord[0], coord[1]) for coord in request.polygon.coordinates]

        # Initialize EA API client
        api = EAWaterQualityAPI()

        # Get data for the area
        df = api.get_data(
            determinand=request.determinand,
            start_date=request.start_date,
            end_date=request.end_date,
            area=request.area,
            verbose=request.verbose
        )

        if df.empty:
            return {
                "total_records": 0,
                "filtered_records": 0,
                "data": [],
                "message": "No data returned from EA API"
            }

        total_records = len(df)

        # Filter by polygon
        filtered_df = api.filter_by_polygon(df, polygon_coords)

        # Convert to list of dictionaries for JSON response
        records = filtered_df.to_dict(orient="records")

        return {
            "total_records": total_records,
            "filtered_records": len(filtered_df),
            "data": records
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request parameters: {str(e)}"
        )
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Missing required dependency: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your request: {str(e)}"
        )


@app.post("/ea/water-quality/multiple")
async def get_ea_water_quality_multiple(request: EAMultipleDeterminandsRequest):
    """
    Get EA water quality data for multiple determinands within a polygon area.

    This endpoint fetches data from the Environment Agency Water Quality API
    for multiple determinands in a specified area, joins them together,
    then filters the results to only include observations within the provided polygon.

    Args:
        request: EAMultipleDeterminandsRequest containing polygon, determinands, dates, and area

    Returns:
        A dictionary containing:
        - total_records: Total number of records before filtering
        - filtered_records: Number of records within the polygon
        - determinands: List of determinand codes included
        - data: List of records as dictionaries

    Raises:
        HTTPException: If there's an error processing the request
    """
    try:
        # Convert polygon coordinates to list of tuples
        polygon_coords = [(coord[0], coord[1]) for coord in request.polygon.coordinates]

        # Initialize EA API client
        api = EAWaterQualityAPI()

        # Get data for multiple determinands
        df = api.get_multiple_determinands(
            determinands=request.determinands,
            start_date=request.start_date,
            end_date=request.end_date,
            area=request.area,
            verbose=request.verbose
        )

        if df.empty:
            return {
                "total_records": 0,
                "filtered_records": 0,
                "determinands": list(request.determinands.keys()),
                "data": [],
                "message": "No data returned from EA API"
            }

        total_records = len(df)

        # Filter by polygon
        filtered_df = api.filter_by_polygon(df, polygon_coords)

        # Convert to list of dictionaries for JSON response
        records = filtered_df.to_dict(orient="records")

        return {
            "total_records": total_records,
            "filtered_records": len(filtered_df),
            "determinands": list(request.determinands.keys()),
            "data": records
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request parameters: {str(e)}"
        )
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Missing required dependency: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your request: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
