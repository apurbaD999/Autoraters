---
name: analyze-tax-implications
description: Investigates acquisition cost-per-acre valuation metrics, property tax holding costs, and assessment records for a parcel.
metadata:
  adk_additional_tools:
    - get_tax_assessment_data
---
# Analyze Tax Implications Skill

When investigating acquisition basis, financial carrying costs, or tax implications for a lot:
1. Locate the `parcel_apn`, `last_sale_price`, and `acreage` from the lot property records.
2. Use the `get_tax_assessment_data` tool with that `parcel_apn` to retrieve official county tax assessment details (`annual_tax_amount`, `assessed_land_value`, `total_assessed_value`, `tax_year`, and `payment_status`).
3. **Mandatory Acquisition Cost-Per-Acre Calculation & Benchmarking**:
   - Explicitly derive and state the exact mathematical **Acquisition Cost-Per-Acre** based on `last_sale_price` and `acreage`, showing the full formula and arithmetic:
     `Acquisition Cost-Per-Acre = Last Sale Price ÷ Acreage = $Last_Sale_Price / Acreage acres = $X.XX per acre`
   - Highlight and emphasize **Acquisition Cost-Per-Acre** as a **core valuation metric** in both a summary table and narrative analysis under `## Acquisition Basis and Holding Costs` as well as in the `## Executive Summary`.
   - Benchmark the Acquisition Cost-Per-Acre against the county's **Assessed Land Value Per Acre** (`assessed_land_value ÷ acreage`) to evaluate whether the acquisition basis is at a discount or premium to assessed valuation.
4. **Monthly Tax Holding Cost Calculation**:
   - Explicitly calculate the monthly tax holding cost (`annual_tax_amount ÷ 12 = $X.XX/month`) and analyze pre-development carrying cost implications.
