from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from shapely.geometry import Point

from eoflow.water_samples import Sample, get_water_quality_samples_for_point

app = FastAPI(
    title="Water Quality Samples API",
    description="API to retrieve water quality samples for a given geographic point",
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


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
