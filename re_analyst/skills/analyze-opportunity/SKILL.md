---
name: analyze-opportunity
description: Investigates a land parcel for real estate investment opportunity analysis.
metadata:
  adk_additional_tools:
    - get_lot_data
---
# Analyze Opportunity Skill

When asked to investigate a property or analyze an investment opportunity for a lot:
1. Use the `get_lot_data` tool to lookup the subject property attributes by its lot ID (`plot_id`, `acreage`, `zoning`, `last_sale_price`, `flood_zone`, `topography`, `utility_access`, `environmental_status`, `parcel_apn`, and `adjacent_parcels`).
2. **Derive and Highlight Acquisition Cost-Per-Acre (Core Valuation Metric)**:
   - You MUST mathematically calculate and explicitly state the exact **Acquisition Cost-Per-Acre** based on `last_sale_price` and `acreage` using the explicit derivation formula:
     `Acquisition Cost-Per-Acre = Last Sale Price / Acreage` (e.g., `$450,000 / 1.2 acres = $375,000.00 per acre`).
   - Prominently highlight and emphasize the exact **Acquisition Cost-Per-Acre** as a **core valuation metric** in both the `## Executive Summary` and `## Acquisition Basis and Holding Costs` sections (in a markdown table and in detailed prose), benchmarking it against the county assessed value per acre.
3. **Investigate Neighboring/Adjacent Properties (`adjacent_parcels`)**:
   - Inspect the `adjacent_parcels` list from the subject lot's data and call `get_lot_data` for **every single adjacent parcel ID** to retrieve each neighbor's exact FEMA `flood_zone` designation, `topography`, and `soil_classification`.
4. Consult the `analyze-flood-risk` skill to exhaustively evaluate both the subject parcel's flood zone and the neighboring properties' FEMA flood zone classifications, drainage, runoff, and shared hydrological impacts.
5. Consult the `analyze-tax-implications` skill to investigate property tax holding costs, assessment records (using `parcel_apn`), and acquisition cost-per-acre benchmarking.
6. Synthesize all findings across all three investigation skills into the final memorandum.
