---
name: dcf-valuation
description: Performs discounted cash flow (DCF) valuation analysis to estimate intrinsic value per share. Triggers when user asks for fair value, intrinsic value, DCF, valuation, "what is X worth", price target, undervalued/overvalued analysis, or wants to compare current price to fundamental value.
---

# DCF Valuation Skill

This skill performs discounted cash flow (DCF) valuation analysis using OpenBB MCP tools. Data providers should prioritize **yfinance** (free) or **fmp** (watch for premium-only features). The widget tools serve as reliable fallbacks.

## Workflow Checklist

Copy and track progress:
```
DCF Analysis Progress:
- [ ] Step 1: Gather financial data
- [ ] Step 2: Calculate FCF growth rate
- [ ] Step 3: Estimate discount rate (WACC)
- [ ] Step 4: Project future cash flows (Years 1-5 + Terminal)
- [ ] Step 5: Calculate present value and fair value per share
- [ ] Step 6: Run sensitivity analysis
- [ ] Step 7: Validate results
- [ ] Step 8: Present results with caveats
```

## Step 1: Gather Financial Data

Use these OpenBB MCP tools in parallel where possible:

### 1.1 Cash Flow History
**Tool:** `openbb-mcp\_equity\_fundamental\_cash`

**Parameters:**
```json
{
  "provider": "yfinance",
  "symbol": "[TICKER]",
  "period": "annual",
  "limit": 5
}
```

**Fallback (if yfinance fails):** `openbb\_widget\_equity\_fundamental\_cash\_yfinance\_obb`

**Extract:** `free\_cash\_flow` (may be labeled as `freeCashFlow`), `operating\_cash\_flow` (`operatingCashFlow`), `capital\_expenditure` (`capitalExpenditure`)

**Calculate FCF if missing:** `operating\_cash\_flow - capital\_expenditure`

---

### 1.2 Balance Sheet Data
**Tool:** `openbb-mcp\_equity\_fundamental\_balance`

**Parameters:**
```json
{
  "provider": "yfinance",
  "symbol": "[TICKER]",
  "period": "annual",
  "limit": 1
}
```

**Extract:** `total\_debt` (or `totalDebt`), `cash\_and\_equivalents` (or `cashAndCashEquivalents`), `short\_term\_investments` (if available)

---

### 1.3 Financial Metrics
**Tool:** `openbb-mcp\_equity\_fundamental\_metrics`

**Parameters:**
```json
{
  "provider": "yfinance",
  "symbol": "[TICKER]"
}
```

**Extract:** `market\_cap`, `free\_cash\_flow\_per\_share`, `revenue\_growth`, `return\_on\_equity` (as proxy for ROIC)

---

### 1.4 Financial Ratios (for ROIC and Debt/Equity)
**Tool:** `openbb-mcp\_equity\_fundamental\_ratios`

**Note:** Only **fmp** and **intrinio** providers available (no yfinance). Use with caution—some FMP features may be premium.

**Parameters:**
```json
{
  "provider": "fmp",
  "symbol": "[TICKER]",
  "period": "annual",
  "ttm": "only"
}
```

**Extract:** `return\_on\_invested\_capital` (ROIC), `debt\_to\_equity` (or `debtEquityRatio`)

**Fallback for D/E:** Calculate from balance sheet: `total\_debt / total\_stockholders\_equity`

---

### 1.5 Company Profile (Sector for WACC)
**Tool:** `openbb-mcp\_equity\_profile`

**Parameters:**
```json
{
  "provider": "yfinance",
  "symbol": "[TICKER]"
}
```

**Extract:** `sector`, `industry`, `market\_cap`, `shares\_outstanding` (if available)
**Use:** Determine appropriate WACC range from [sector-wacc.md](references/sector-wacc.md)

---

### 1.6 Share Statistics (for Outstanding Shares)
**Tool:** `openbb\_widget\_share\_statistics`

**Parameters:**
```json
{
  "symbol": "[TICKER]"
}
```

**Extract:** `shares\_outstanding` (or `SharesOutstanding`, `impliedSharesOutstanding`)

---

### 1.7 Current Price
**Tool:** `openbb-mcp\_equity\_price\_quote`

**Parameters:**
```json
{
  "provider": "yfinance",
  "symbol": "[TICKER]"
}
```

**Extract:** `price` (or `regularMarketPrice`, `currentPrice`)

---

### 1.8 Analyst Estimates (for Growth Cross-Validation)
**Tool:** `openbb-mcp\_equity\_estimates\_forward\_eps` OR `openbb-mcp\_equity\_estimates\_consensus`

**For forward EPS (FMP provider):**
```json
{
  "provider": "fmp",
  "symbol": "[TICKER]",
  "fiscal\_period": "annual",
  "limit": 3
}
```

**For consensus (yfinance available):**
```json
{
  "provider": "yfinance",
  "symbol": "[TICKER]"
}
```

**Extract:** Forward EPS estimates, implied growth rates, consensus price target (if available)

---

### 1.9 Risk-Free Rate (for WACC)
**Tool:** `openbb-mcp\_fixedincome\_government\_treasury\_rates`

**Parameters:**
```json
{
  "provider": "federal\_reserve"
}
```

**Extract:** 10-year Treasury yield (use as risk-free rate baseline)

**Fallback:** Assume 4% risk-free rate

---

## Step 2: Calculate FCF Growth Rate

Calculate 5-year FCF CAGR from cash flow history.

**Cross-validate with:** `free_cash_flow_growth` (YoY), `revenue_growth`, analyst EPS growth

**Growth rate selection:**
- Stable FCF history → Use CAGR with 10-20% haircut
- Volatile FCF → Weight analyst estimates more heavily
- **Cap at 15%** (sustained higher growth is rare)

## Step 3: Estimate Discount Rate (WACC)

**Use the `sector` from company facts** to select the appropriate base WACC range from [sector-wacc.md](references/sector-wacc.md).

**Default assumptions:**
- Risk-free rate: 4%
- Equity risk premium: 5-6%
- Cost of debt: 5-6% pre-tax (~4% after-tax at 30% tax rate)

Calculate WACC using `debt_to_equity` for capital structure weights.

**Reasonableness check:** WACC should be 2-4% below `return_on_invested_capital` for value-creating companies.

**Sector adjustments:** Apply adjustment factors from [sector-wacc.md](references/sector-wacc.md) based on company-specific characteristics.

## Step 4: Project Future Cash Flows

**Years 1-5:** Apply growth rate with 5% annual decay (multiply growth rate by 0.95, 0.90, 0.85, 0.80 for years 2-5). This reflects competitive dynamics.

**Terminal value:** Use Gordon Growth Model with 2.5% terminal growth (GDP proxy).

## Step 5: Calculate Present Value

Discount all FCFs → sum for Enterprise Value → subtract Net Debt → divide by `outstanding_shares` for fair value per share.

## Step 6: Sensitivity Analysis

Create 3×3 matrix: WACC (base ±1%) vs terminal growth (2.0%, 2.5%, 3.0%).

## Step 7: Validate Results

Before presenting, verify these sanity checks:

1. **EV comparison**: Calculated EV should be within 30% of reported `enterprise_value`
   - If off by >30%, revisit WACC or growth assumptions

2. **Terminal value ratio**: Terminal value should be 50-80% of total EV for mature companies
   - If >90%, growth rate may be too high
   - If <40%, near-term projections may be aggressive

3. **Per-share cross-check**: Compare to `free_cash_flow_per_share × 15-25` as rough sanity check

If validation fails, reconsider assumptions before presenting results.

## Step 8: Output Format

Present a structured summary including:
1. **Valuation Summary**: Current price vs. fair value, upside/downside percentage
2. **Key Inputs Table**: All assumptions with their sources
3. **Projected FCF Table**: 5-year projections with present values
4. **Sensitivity Matrix**: 3×3 grid varying WACC (±1%) and terminal growth (2.0%, 2.5%, 3.0%)
5. **Caveats**: Standard DCF limitations plus company-specific risks

### 8.1 HTML Report Format (Optional)

If user requests an HTML report, use the style template at [html-template-style.md](references/html-template-style.md).

**Template sections to populate:**

| Template Section | DCF Content |
|-----------------|-------------|
| `.company-name` | Company name and ticker |
| `.subtitle` | Valuation date and analysis type (DCF) |
| `.section-header` | Valuation Summary, Key Assumptions, FCF Projections, Sensitivity Analysis |
| `.body-text` | Valuation narrative and interpretation |
| `.kfd-table` | Key Financial Data: Current price, Fair value, Upside/Downside %, WACC, Terminal growth rate |
| `.fin-table` | 5-year FCF Projections table with discount factors and present values |
| `.fin-section-title` | "Financial Projections", "Sensitivity Analysis" |
| `.tp-history-table` | Sensitivity matrix (WACC rows, Terminal growth columns, Fair value results) |
| `.rating-box` | DCF methodology and assumptions caveats |
| `.company-logo` | Replace `[LOGO]` with company identifier or analyst initials |

**Steps:**
1. Open [html-template-style.md](references/html-template-style.md) as the base
2. Populate `[Company]` placeholders with actual company name
3. Insert narrative analysis in `.body-text` sections
4. Embed tables using `.fin-table` and `.tp-history-table` classes
5. Include SVG charts (valuation trend, sensitivity heatmap) in `.chart-container` divs

---

## Provider Selection Guide

| Data Type | Primary | Fallback | Notes |
|-----------|---------|----------|-------|
| Cash Flow | yfinance | widget | Widget uses yfinance under hood |
| Balance Sheet | yfinance | fmp | Check for premium restrictions |
| Metrics | yfinance | fmp | |
| Ratios | fmp | calculate manually | yfinance unavailable; watch for FMP premium |
| Profile | yfinance | fmp | |
| Price | yfinance | fmp | |
| Estimates | fmp | yfinance consensus | FMP may have premium limits |
| Treasury Rates | federal\_reserve | assumed 4% | Free from Fed |
| Share Stats | widget | profile data | Widget most reliable for shares |
