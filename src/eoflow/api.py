import json
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from shapely.geometry import Point

from eoflow.cs import CSDataAPI
from eoflow.ea import EAWaterQualityAPI
from eoflow.water_samples import Sample, get_water_quality_samples_for_point

app = FastAPI(
    title="Water Quality API",
    description="API to retrieve water quality samples from EA monitoring data and citizen scientist sources",
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
        min_length=3,
    )

    class Config:
        json_schema_extra = {
            "example": {
                "coordinates": [
                    [-4.5, 50.3],
                    [-4.5, 51.2],
                    [-3.0, 51.2],
                    [-3.0, 50.3],
                    [-4.5, 50.3],
                ]
            }
        }


class EAWaterQualityRequest(BaseModel):
    """Request model for EA water quality data"""

    polygon: PolygonRequest = Field(
        ..., description="Polygon defining the area of interest"
    )
    determinand: str = Field(
        ..., description="Determinand code (e.g., '0076' for temperature)"
    )
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    area: str = Field(
        ..., description="Precanned area code (e.g., 'environment_agency,SWX')"
    )
    verbose: bool = Field(
        default=False, description="Whether to show verbose logging"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "polygon": {
                    "coordinates": [
                        [-4.5, 50.3],
                        [-4.5, 51.2],
                        [-3.0, 51.2],
                        [-3.0, 50.3],
                        [-4.5, 50.3],
                    ]
                },
                "determinand": "0076",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "area": "environment_agency,SWX",
                "verbose": False,
            }
        }


class EAMultipleDeterminandsRequest(BaseModel):
    """Request model for multiple EA determinands"""

    polygon: PolygonRequest = Field(
        ..., description="Polygon defining the area of interest"
    )
    determinands: Dict[str, str] = Field(
        ..., description="Dictionary mapping determinand codes to column names"
    )
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    area: str = Field(
        ..., description="Precanned area code (e.g., 'environment_agency,SWX')"
    )
    verbose: bool = Field(
        default=False, description="Whether to show verbose logging"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "polygon": {
                    "coordinates": [
                        [-4.5, 50.3],
                        [-4.5, 51.2],
                        [-3.0, 51.2],
                        [-3.0, 50.3],
                        [-4.5, 50.3],
                    ]
                },
                "determinands": {"0076": "Temperature", "0077": "Conductivity"},
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "area": "environment_agency,SWX",
                "verbose": False,
            }
        }


class CSDataRequest(BaseModel):
    """Request model for CS (Citizen Scientist) data"""

    polygon: PolygonRequest = Field(
        ..., description="Polygon defining the area of interest"
    )
    base_url: str = Field(..., description="ArcGIS FeatureServer base URL")
    layer_id: int = Field(
        default=0, description="Layer ID within the FeatureServer"
    )
    where: str = Field(
        default="1=1", description="SQL WHERE clause for filtering"
    )
    start_date: Optional[str] = Field(
        None, description="Start date in YYYY-MM-DD format"
    )
    end_date: Optional[str] = Field(
        None, description="End date in YYYY-MM-DD format"
    )
    date_field: str = Field(
        default="sample_date", description="Name of the date field"
    )
    verbose: bool = Field(
        default=False, description="Whether to show verbose logging"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "polygon": {
                    "coordinates": [
                        [-4.5, 50.3],
                        [-4.5, 51.2],
                        [-3.0, 51.2],
                        [-3.0, 50.3],
                        [-4.5, 50.3],
                    ]
                },
                "base_url": "https://services.arcgis.com/xxx/FeatureServer",
                "layer_id": 0,
                "where": "1=1",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "date_field": "sample_date",
                "verbose": False,
            }
        }


class CombinedWaterQualityRequest(BaseModel):
    """Request model for combined EA and CS data"""

    polygon: PolygonRequest = Field(
        ..., description="Polygon defining the area of interest"
    )

    # EA parameters
    ea_determinand: Optional[str] = Field(
        None, description="EA determinand code (e.g., '0076')"
    )
    ea_area: Optional[str] = Field(None, description="EA precanned area code")

    # CS parameters
    cs_base_url: Optional[str] = Field(
        None, description="CS ArcGIS FeatureServer base URL"
    )
    cs_layer_id: int = Field(default=0, description="CS layer ID")
    cs_where: str = Field(default="1=1", description="CS WHERE clause")
    cs_date_field: str = Field(
        default="sample_date", description="CS date field name"
    )

    # Common parameters
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    verbose: bool = Field(
        default=False, description="Whether to show verbose logging"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "polygon": {
                    "coordinates": [
                        [-4.5, 50.3],
                        [-4.5, 51.2],
                        [-3.0, 51.2],
                        [-3.0, 50.3],
                        [-4.5, 50.3],
                    ]
                },
                "ea_determinand": "0076",
                "ea_area": "environment_agency,SWX",
                "cs_base_url": "https://services.arcgis.com/xxx/FeatureServer",
                "cs_layer_id": 0,
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "verbose": False,
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
            "/cs/water-quality": "POST - Get CS (citizen scientist) data within a polygon",
            "/combined/water-quality": "POST - Get combined EA and CS water quality data",
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
        polygon_coords = [
            (coord[0], coord[1]) for coord in request.polygon.coordinates
        ]

        # Initialize EA API client
        api = EAWaterQualityAPI()

        # Get data for the area
        df = api.get_data(
            determinand=request.determinand,
            start_date=request.start_date,
            end_date=request.end_date,
            area=request.area,
            verbose=request.verbose,
        )

        if df.empty:
            return {
                "total_records": 0,
                "filtered_records": 0,
                "data": [],
                "message": "No data returned from EA API",
            }

        total_records = len(df)

        # Filter by polygon
        filtered_df = api.filter_by_polygon(df, polygon_coords)

        # Convert to list of dictionaries for JSON response
        records = filtered_df.to_dict(orient="records")

        return {
            "total_records": total_records,
            "filtered_records": len(filtered_df),
            "data": records,
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid request parameters: {str(e)}"
        )
    except ImportError as e:
        raise HTTPException(
            status_code=500, detail=f"Missing required dependency: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your request: {str(e)}",
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
        polygon_coords = [
            (coord[0], coord[1]) for coord in request.polygon.coordinates
        ]

        # Initialize EA API client
        api = EAWaterQualityAPI()

        # Get data for multiple determinands
        df = api.get_multiple_determinands(
            determinands=request.determinands,
            start_date=request.start_date,
            end_date=request.end_date,
            area=request.area,
            verbose=request.verbose,
        )

        if df.empty:
            return {
                "total_records": 0,
                "filtered_records": 0,
                "determinands": list(request.determinands.keys()),
                "data": [],
                "message": "No data returned from EA API",
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
            "data": records,
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid request parameters: {str(e)}"
        )
    except ImportError as e:
        raise HTTPException(
            status_code=500, detail=f"Missing required dependency: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your request: {str(e)}",
        )


@app.post("/cs/water-quality")
async def get_cs_water_quality(request: CSDataRequest):
    """
    Get CS (Citizen Scientist) water quality data within a polygon area.

    This endpoint fetches data from an ArcGIS FeatureServer containing
    citizen scientist observations, then filters the results to only
    include observations within the provided polygon.

    Args:
        request: CSDataRequest containing polygon, base URL, and filter parameters

    Returns:
        A dictionary containing:
        - total_records: Total number of records before filtering
        - filtered_records: Number of records within the polygon
        - data_source: "CS" to indicate citizen scientist data
        - data: List of records as dictionaries

    Raises:
        HTTPException: If there's an error processing the request
    """
    try:
        # Convert polygon coordinates to list of tuples
        polygon_coords = [
            (coord[0], coord[1]) for coord in request.polygon.coordinates
        ]

        # Initialize CS API client
        api = CSDataAPI(base_url=request.base_url, layer_id=request.layer_id)

        # Get data from the FeatureServer
        df = api.get_data(
            where=request.where,
            start_date=request.start_date,
            end_date=request.end_date,
            date_field=request.date_field,
            verbose=request.verbose,
        )

        if df.empty:
            return {
                "total_records": 0,
                "filtered_records": 0,
                "data_source": "CS",
                "data": [],
                "message": "No data returned from CS API",
            }

        total_records = len(df)

        # Filter by polygon
        filtered_df = api.filter_by_polygon(df, polygon_coords)

        # Add data source column
        filtered_df = filtered_df.copy()
        filtered_df["data_source"] = "CS"

        # Convert to list of dictionaries for JSON response
        records = filtered_df.to_dict(orient="records")

        return {
            "total_records": total_records,
            "filtered_records": len(filtered_df),
            "data_source": "CS",
            "data": records,
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid request parameters: {str(e)}"
        )
    except ImportError as e:
        raise HTTPException(
            status_code=500, detail=f"Missing required dependency: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your request: {str(e)}",
        )


@app.post("/combined/water-quality")
async def get_combined_water_quality(request: CombinedWaterQualityRequest):
    """
    Get combined EA and CS water quality data within a polygon area.

    This endpoint fetches data from both Environment Agency and Citizen Scientist
    sources, filters by the provided polygon, and combines the results with a
    'data_source' field to indicate the origin of each record.

    Args:
        request: CombinedWaterQualityRequest with polygon and parameters for both sources

    Returns:
        A dictionary containing:
        - ea_records: Number of EA records
        - cs_records: Number of CS records
        - total_combined_records: Total number of records from both sources
        - data: List of all records with 'data_source' field ('EA' or 'CS')

    Raises:
        HTTPException: If there's an error processing the request
    """
    try:
        # Convert polygon coordinates to list of tuples
        polygon_coords = [
            (coord[0], coord[1]) for coord in request.polygon.coordinates
        ]

        combined_data = []
        ea_count = 0
        cs_count = 0
        errors = []

        # Fetch EA data if parameters provided
        if request.ea_determinand and request.ea_area:
            try:
                ea_api = EAWaterQualityAPI()
                ea_df = ea_api.get_data(
                    determinand=request.ea_determinand,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    area=request.ea_area,
                    verbose=request.verbose,
                )

                if not ea_df.empty:
                    # Filter by polygon
                    ea_filtered = ea_api.filter_by_polygon(
                        ea_df, polygon_coords
                    )

                    if not ea_filtered.empty:
                        # Add data source column
                        ea_filtered = ea_filtered.copy()
                        ea_filtered["data_source"] = "EA"
                        ea_count = len(ea_filtered)
                        combined_data.append(ea_filtered)
            except Exception as e:
                errors.append(f"EA API error: {str(e)}")

        # Fetch CS data if parameters provided
        if request.cs_base_url:
            try:
                cs_api = CSDataAPI(
                    base_url=request.cs_base_url, layer_id=request.cs_layer_id
                )
                cs_df = cs_api.get_data(
                    where=request.cs_where,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    date_field=request.cs_date_field,
                    verbose=request.verbose,
                )

                if not cs_df.empty:
                    # Filter by polygon
                    cs_filtered = cs_api.filter_by_polygon(
                        cs_df, polygon_coords
                    )

                    if not cs_filtered.empty:
                        # Add data source column
                        cs_filtered = cs_filtered.copy()
                        cs_filtered["data_source"] = "CS"
                        cs_count = len(cs_filtered)
                        combined_data.append(cs_filtered)
            except Exception as e:
                errors.append(f"CS API error: {str(e)}")

        # Combine all data
        if combined_data:
            combined_df = pd.concat(combined_data, ignore_index=True)
            # Convert to dict with NaN handling for JSON serialization
            records_json = combined_df.to_json(orient="records")
            records = json.loads(records_json)
        else:
            records = []

        response = {
            "ea_records": ea_count,
            "cs_records": cs_count,
            "total_combined_records": ea_count + cs_count,
            "data": records,
        }

        if errors:
            response["errors"] = errors

        if not records and not errors:
            response["message"] = "No data returned from any source"

        return response

    except ValueError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid request parameters: {str(e)}"
        )
    except ImportError as e:
        raise HTTPException(
            status_code=500, detail=f"Missing required dependency: {str(e)}"
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
