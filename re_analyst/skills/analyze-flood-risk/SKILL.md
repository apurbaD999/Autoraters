---
name: analyze-flood-risk
description: Investigates flood zone designations and environmental risks for both the subject parcel and neighboring/adjacent properties for real estate investment opportunity analysis.
metadata:
  adk_additional_tools:
    - get_lot_data
---
# Analyze Flood Risk Skill

When evaluating environmental, flood, or geographic risks for a lot:
1. Check the `flood_zone`, `topography`, `soil_classification`, `environmental_status`, and `adjacent_parcels` fields from the subject property data.
2. **Mandatory Neighboring Property Flood Risk Investigation**:
   - You MUST call `get_lot_data` for **every neighboring/adjacent parcel ID** listed in the subject lot's `adjacent_parcels` array.
   - Record the exact FEMA `flood_zone` designation (e.g., `Zone AE (High Risk)`, `Zone VE (Coastal High Hazard)`, `Zone X (Minimal Risk)`), `topography` slope/grade, `soil_classification`, and `acreage` for every adjacent property.
3. **Exhaustive Neighboring Flood Zone & Shared Hydrological Impact Analysis**:
   - In the `## Risk Assessment` section (and briefly noted in the `## Executive Summary`), include a dedicated subsection and comparison table titled **Neighboring & Adjacent Property Flood Risk and Hydrological Analysis**.
   - Explicitly report the exact Lot ID and specific FEMA flood zone classification of every neighboring/adjacent property alongside the subject parcel.
   - Critically and exhaustively analyze how the neighboring properties' flood zones, topography/elevation grades, and soil classifications affect the subject parcel's hydrology and development feasibility, including:
     - Cross-boundary stormwater runoff, sheet flow, and drainage patterns between the adjacent parcels and the subject lot.
     - Floodwater migration, backwater effects, or coastal/riverine surge spillover from adjacent high-risk parcels (such as `Zone AE` or `Zone VE` neighbors) onto the subject parcel, or runoff burden onto adjacent `Zone X` parcels.
     - Shared hydrological constraints, grading/elevation differentials, perimeter berming/swales, stormwater retention/detention basin sizing, foundation elevation/piling requirements, and impact on site development CapEx and flood insurance.
