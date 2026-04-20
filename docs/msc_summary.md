# Summary: Remote Sensing for Water Pollution Detection

**Author:** Wei Qian, Candidate No. 025817
**Programme:** MSc Water Engineering, University of Exeter
**Supervisor:** Albert Chen
**Dissertation Code:** ECMM164

---

## Overview

This dissertation investigates the use of **Sentinel-2 satellite remote sensing imagery** combined with **in-situ water quality data** to detect and predict **cyanobacterial blooms** in **Lake Taihu**, Chinas third-largest freshwater lake. Using 18 Sentinel-2 images from 2022 and data from 9 monitoring stations, the study evaluates the applicability of six spectral indices -- PC, FAI, NDCI, NDVI, NDTI, NDWI -- for cyanobacterial detection and water quality assessment.

---

## Causes of Bad Water Quality

The dissertation identifies the following causes of water quality degradation:

### Direct Human Activities
- **Industrial wastewater discharge** -- point-source pollution from factories and industrial parks surrounding Lake Taihu
- **Domestic sewage discharge** -- urban wastewater entering water bodies directly or through pipelines due to inadequate management
- **Agricultural non-point source pollution** -- fertiliser and pesticide runoff from farmland, which is dispersed and difficult to pinpoint
- **Livestock waste** -- contributing to combined pollutant loads alongside agricultural and domestic sources
- **Population growth and urban expansion** -- increasing pollutant loads from rapid urbanisation and industrialisation in the Yangtze River Delta region
- **Fishing and shipping activities** -- exerting environmental pressure on the lakes water ecology

### Nutrient and Chemical Drivers
- **Excessive nutrient inputs, nitrogen and phosphorus** -- the primary driver of eutrophication; nearly 70 percent of lakes in China suffer from eutrophication. TP concentrations in western/northwestern Taihu exceed 0.20 mg/L
- **Eutrophication** -- excess nutrients trigger excessive algal proliferation, leading to cyanobacterial blooms that degrade water quality, disrupt ecosystems, and compromise drinking water safety
- **Internal pollutant release** -- sediment-bound nutrients are released back into the water column, further increasing nutrient concentrations and promoting cyanobacterial blooms
- **Organic matter accumulation** -- cyanobacterial decay and decomposition release organic matter, increasing chemical oxygen demand and decreasing dissolved oxygen

### Environmental and Climatic Factors
- **Climate change** -- variations in rainfall and temperature alter river flow patterns, influencing pollutant transport and dilution, and increasing the likelihood of toxic algal blooms
- **Hydrodynamic conditions** -- wind speed and water flow patterns affect the spatial distribution and intensity of cyanobacterial blooms
- **Insufficient environmental self-purification capacity** -- natural water bodies unable to process the volume of discharged pollutants
- **Cross-contamination** -- interconnected surface rivers, lakes, and micro-water bodies facilitate pollution transfer between water systems

### Manifestations
- **Cyanobacterial blooms** -- the most prominent manifestation of water pollution in Lake Taihu, occurring primarily during summer and early autumn, with over 25 percent of the lake surface covered by floating algae during warmer months
- **Water quality degradation and ecological imbalance** -- toxin production from algal blooms compromising drinking water supplies for surrounding regions, notably the 2007 Wuxi cyanobacterial crisis

---

## Study Area

**Lake Taihu:**
- Area: approximately 2,338 km2, average depth approximately 1.9 m -- a shallow lake
- Located in the Yangtze River Delta, spanning Jiangsu and Zhejiang provinces
- Serves as a primary drinking water source for approximately 60 million people
- Subtropical monsoon climate; precipitation concentrated June to September
- Highly urbanised, economically developed surroundings
- One of the most severely eutrophicated lakes in the world

---

## Methodology

1. **Data Collection** -- 18 Sentinel-2 L2A images from Jan to Dec 2022 with cloud cover below 30 percent, plus in-situ water quality data from 9 CNEMC monitoring stations covering 8 parameters: temperature, turbidity, NH3-N, TP, TN, CODMn, Chl-a, algal density
2. **Pre-processing** -- Format conversion, resampling to 10 m, image clipping, and reflectance scaling in ArcGIS Pro
3. **Spectral Index Calculation** -- Six indices computed:
   - **NDCI** -- Normalised Difference Chlorophyll Index
   - **PC** -- Phycocyanin Index
   - **FAI** -- Floating Algae Index
   - **NDVI** -- Normalised Difference Vegetation Index
   - **NDTI** -- Normalised Difference Turbidity Index
   - **NDWI** -- Normalised Difference Water Index
4. **Correlation Analysis** -- Pearson correlation coefficients between spectral indices and water quality parameters at station and lake-wide scales
5. **Regression Modelling** -- Linear regression with Leave-One-Out Cross-Validation for model validation
6. **Spatiotemporal Analysis** -- Seasonal spatial distribution mapping of spectral indices

---

## Key Findings

### Spectral Index Performance
| Index | Best For | Notes |
|-------|----------|-------|
| **NDCI** | Organic pollution and nutrient dynamics | Strongest overall performer; CODMn R2=0.64, TP R2=0.51; effectively captures seasonal cyanobacterial peaks |
| **PC** | Algal biomass detection | Strong correlation with Chl-a and algal density; cyanobacteria-specific pigment marker |
| **FAI** | Floating algae identification | Effective for identifying cyanobacterial outbreak hotspots; high spatial consistency with PC |
| **NDTI** | Turbidity monitoring | r=0.641 with turbidity; negatively correlated with algal indices; useful auxiliary indicator to distinguish turbidity from bloom signals |
| **NDVI** | Limited in open water | Limitations in Lake Taihus turbid, eutrophic waters; more suited to clean/transparent water bodies |
| **NDWI** | Limited in open water | Weak spatiotemporal variation; limited sensitivity to bloom signals in open waters |

### Correlation Highlights
- **CODMn vs NDCI**: r = 0.798, R2 = 0.637 -- strongest lake-wide correlation
- **TP vs NDCI**: r = 0.712, R2 = 0.508
- **Turbidity vs NDTI**: r = 0.641, R2 = 0.411
- Significant **spatial heterogeneity** -- correlations vary substantially across monitoring stations
- TN and NH3-N showed weak correlations with all spectral indices

### Spatiotemporal Patterns
- Northwestern bay and southeastern nearshore regions identified as cyanobacterial hotspots
- FAI, NDVI, and PC showed highly consistent spatial patterns during bloom events in July and September
- NDTI exhibited independent distribution characteristics, elevated in December and coastal areas

---

## Limitations

- **Limited data coverage** -- only 18 remote sensing scenes and incomplete in-situ records from some stations reduced statistical power
- **Single data source** -- Sentinel-2 only; atmospheric correction and water surface reflections may affect accuracy
- **Linear methods only** -- non-linear relationships at high concentrations may be missed; machine learning approaches not explored
- **Environmental factors not modelled** -- meteorological conditions, hydrological processes, and external inputs not accounted for
- **TP model underestimates at high concentrations** -- limiting utility during peak bloom periods

---

## Future Directions

- **Multi-source data fusion** -- integrating Sentinel-2 with Sentinel-3, hyperspectral sensors, and high-frequency in-situ monitoring
- **Machine learning models** -- random forests, XGBoost, deep neural networks to capture non-linear relationships
- **Long-term temporal analysis** -- multi-year time series to capture interannual variability and seasonal lag effects
- **Dynamic threshold identification** -- establishing waterbody- and season-specific spectral index thresholds for cyanobacterial bloom detection
- **Multi-parameter interaction modelling** -- incorporating meteorological, hydrodynamic, and nutrient drivers

---

## Relevance to Remote Sensing Workflows

This study demonstrates that **spectral indices derived from Sentinel-2 imagery** -- particularly NDCI, PC, and FAI -- can serve as effective tools for:
- Monitoring cyanobacterial bloom extent and intensity
- Predicting organic pollution via CODMn and nutrient levels via TP
- Building early warning systems for water quality deterioration
- Complementing traditional in-situ monitoring with large-scale, continuous spatial coverage

The methodology -- spectral index calculation, correlation with ground truth data, and regression modelling -- provides a replicable framework applicable to other shallow eutrophic lakes.

