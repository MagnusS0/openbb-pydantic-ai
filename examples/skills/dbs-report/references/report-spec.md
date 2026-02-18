# DBS US Equity Research Report -- Complete Formatting Specification

This document defines every visual and structural element of a DBS Group Research US Equity Research report so that a generated report is indistinguishable from the originals.

---

## 1. DOCUMENT METADATA

| Property | Value |
|----------|-------|
| Page size | A4 portrait (210mm x 297mm) |
| Margins | Top: 15mm, Bottom: 25mm, Left: 20mm, Right: 15mm |
| Total pages | 7 (3 content + 4 boilerplate) |
| Font family | Sans-serif throughout (Calibri, Segoe UI, or Arial), including banner |

---

## 2. RECURRING PAGE ELEMENTS (Every Page)

### 2.1 Top-Right Header
- Text: "DBS Group Research"
- Font: ~9pt, regular weight, right-aligned
- Position: Top-right corner, inside top margin

### 2.2 Bottom Disclaimer Footer
- Font: ~7pt, regular weight
- 3 lines of text:
  ```
  Disclaimer: The information contained in this document is intended only for use by the person to whom it has been delivered and should not be disseminated
  or distributed to third parties without our prior written consent. DBS accepts no liability whatsoever with respect to the use of this document or its contents.
  Please refer to Disclaimer found at the end of this document
  ```

### 2.3 Bottom-Right Logo
- DBS logo: Red diamond star icon (&#10070; / ❖) followed by "DBS" in bold
- Position: Bottom-right corner, below the disclaimer
- Colors: Red (#ED1C24) for diamond star, black for "DBS" text
- HTML: `<span style="color:#ED1C24">&#10070;</span><span style="font-weight:bold">DBS</span>`

---

## 3. PAGE 1 -- COVER PAGE

### 3.1 Layout
Two-column layout:
- **Left column**: ~60% width -- contains all narrative text
- **Right column**: ~40% width -- contains Key Financial Data table and indexed chart

### 3.2 Title Banner
- **Shape**: Solid black (#000000) filled rectangle
- **Width**: Spans the full page width (100%)
- **Height**: ~30px
- **Content**: "US EQUITY RESEARCH" in white, uppercase, sans-serif, ~18-20pt
- **Alignment**: Left-aligned text within the banner, banner starts at left margin

### 3.3 Date Line
- Position: Right-aligned, immediately below the banner (right edge of banner)
- Font: ~10pt, regular weight
- Format: `DD Month YYYY`

### 3.4 Company Name
- Position: Left-aligned, below the banner (with ~10px gap)
- Font: ~18pt, regular weight (NOT bold), serif
- Single line

### 3.5 Subtitle / Tagline
- Position: Immediately below company name
- Font: ~10pt, regular weight, serif
- Single line, concise investment thesis summary

### 3.6 Analyst Attribution (Right Column)
- Position: Right column, vertically aligned with company name
- Format:
  ```
  Analyst:                    (or "Analysts" if multiple)
  Name | email@dbs.com
  ```
- Font: ~9pt

### 3.7 Section Headers (Left Column)
All narrative sections use identical header formatting:
- Font: ~11pt, regular weight
- Followed by a thin horizontal rule (1px, light gray #CCCCCC) spanning the left column width
- ~8px gap before body text begins
- Sections appear in this exact order:
  1. "Company Overview"
  2. "Investment Overview"
  3. "Risks"

### 3.8 Body Text (Left Column)
- Font: ~9.5pt serif, regular weight
- Line height: ~1.3
- Alignment: **Fully justified**
- Paragraph spacing: ~6px between paragraphs

### 3.9 Bold Lead-In Sentences
Each paragraph in "Investment Overview" and "Risks" begins with a bold **and underlined** opening sentence:
- The bold+underlined portion is the first complete sentence (ending at the first period)
- After the period, text continues in regular weight (no underline) on the same line
- This bold+underlined sentence functions as a topic sentence / mini-headline
- CSS: `font-weight: bold; text-decoration: underline;`
- Examples (bold+underline shown):
  - "**<u>iPhone specification upgrades to drive premium model mix higher.</u>** We expect the launches..."
  - "**<u>Leveraging scale for long-term market share gains.</u>** Blackstone Inc. stands as..."
  - "**<u>Delays in Apple Intelligence launch could impact growth.</u>** Apple's in-house LLM..."

### 3.10 Key Financial Data Table (Right Column)
- Title: "Key Financial Data" (~11pt, regular weight, no underline)
- Table style: Two columns, thin horizontal rules as row separators
- Left column (labels): Left-aligned, ~9pt
- Right column (values): Right-aligned, ~9pt
- No vertical borders
- Row items in exact order:

| Row | Label | Example Value |
|-----|-------|---------------|
| 1 | Bloomberg Ticker | AAPL US |
| 2 | Sector | Information Technology |
| 3 | Share Price (USD) | 210.02 |
| 4 | DBS Rating | HOLD |
| 5 | 12-mth Target Price (USD) | 210.00 |
| 6 | Market Cap (USDbn) | 3136.8 |
| 7 | Volume (mn shares) | 48,068.1 |
| 8 | Free float (%) | 97.9 |
| 9 | Dividend yield (%) | 0.5 |
| 10 | Net Debt to Equity (%) | -66.0 |
| 11 | Fwd. P/E (x) | 29.3 |
| 12 | P/Book (x) | 47.0 |
| 13 | ROE (%) | 138.0 |

- Below table (italicized): *Closing Price as of DD Mon YYYY*
- Source line: "Source: Bloomberg, DBS"

### 3.11 Indexed Share Price Chart (Right Column)
- Title: "Indexed Share Price vs Composite Index Performance"
- Type: Line chart, two series
- Series 1: Company name (orange/gold line, `#D4A843`)
- Series 2: "S&P 500" (dark gray/black line, `#333333`)
- Y-axis: Indexed values, labeled "(indexed)" at top-left of chart
- X-axis: ~4 year range, labels at 6-month intervals, format "Mon-YY"
- Gridlines: Horizontal only, light gray (`#E0E0E0`)
- Legend: Top of chart area, horizontal layout with colored line samples
- Chart dimensions: ~300px wide x ~200px tall
- Source line below: "Source: Bloomberg"

---

## 4. PAGE 2 -- FINANCIAL DATA TABLES

Full-width layout (no columns). Three tables stacked vertically with ~15px gap between them.

### 4.1 Table Style (Common to All Three)
- No outer borders
- Thin horizontal rules between rows (~0.5px, `#CCCCCC`)
- Header row: Bold text, with a slightly thicker top border
- Column alignment: First column left-aligned (labels), all numeric columns right-aligned
- Font: ~9pt for body, ~9.5pt bold for headers
- Source attribution below each table: "Source: Visible Alpha" (regular weight, NOT italic, ~8pt)

### 4.2 Financial Summary Table

**Title**: "Financial Summary (USDmn)" or with currency in header row

**Column headers**: `FY Dec | FY2022A | FY2023A | FY2024A | FY2025F | FY2026F`

**Rows** (in exact order):

```
Sales
  % y/y                    ← italicized, indented ~15px
Gross Profit
  % y/y
EBITDA
  % y/y
Net Profit (Loss)
  % y/y
FCF
  % y/y
CAPEX
  % y/y
                            ← visual gap (empty row or extra spacing)
EBITDA Margin (%)
Net Margin (%)
ROA (%)
ROE (%)
Tax Rate (%)
```

**Number formatting**:
- Absolute values: Comma-separated thousands (e.g., `394,328`)
- y/y percentages: One decimal place, negatives in parentheses (e.g., `(2.8)`)
- Margin/ratio percentages: One decimal place

### 4.3 Valuation Metrics Table

**Title**: "Valuation Metrics"

**Rows**:
```
P/E
P/B
Dividend Yield
EV/EBITDA (x)
FCF Yield %
```

### 4.4 Credit & Cashflow Metrics Table

**Title**: "Credit & Cashflow Metrics"

**Rows**:
```
Debt / Equity
Net Debt / Equity
Debt / Assets
Net Debt / Assets
EBITDA / Int Exp
Debt / EBITDA
ST Debt / Total Debt (%)
[Cash + CFO] / ST Debt
Receivables Days
Days Payable
Inventory Days
```

---

## 5. PAGE 3 -- TARGET PRICE & RATINGS HISTORY

### 5.1 Section Title
"Target Price & Ratings History" -- bold, with underline rule

### 5.2 Chart (Left ~55%)
- Type: Combined line chart
- Series 1 "Stock Target Change" / "DBSTP": Black step-line or shaded area showing target price levels
- Series 2 "SHARE PRICE": Red/coral line (`#CC3333`) showing actual stock price
- Numbered circle markers (1, 2, 3...) placed at each report date on the price line
- Y-axis: Stock Price (USD), labeled
- X-axis: ~12 month range, monthly labels in "Mon-YY" format (rotated ~45 degrees)
- Legend at top of chart

### 5.3 Companion Table (Right ~45%)
- Columns: `# | Date of Report | Closing Price | 12-m Target Price | Rating`
- Header row: Coral/red background (`#CC3333`) with white text
- Body rows: Standard black text on white
- Date format: "DD Mon'YY" (e.g., "21 Jan'25")
- Each row corresponds to a numbered marker on the chart

### 5.4 Attribution
- "Source: DBS"
- "Analyst: [Name]" or "Analysts: [Name1] / [Name2]"

### 5.5 Rating Definitions Box
- Thin black border box (~1px solid black)
- Opening line: "DBS Group Research recommendations are based on an Absolute Total Return* Rating system, defined as follows:"
- Each rating on its own line, rating name in bold + underline:
  - **STRONG BUY** (>20% total return over the next 3 months, with identifiable share price catalysts within this time frame)
  - **BUY** (>15% total return over the next 12 months for small caps, >10% for large caps)
  - **HOLD** (-10% to +15% total return over the next 12 months for small caps, -10% to +10% for large caps)
  - **FULLY VALUED** (negative total return, i.e., > -10% over the next 12 months)
  - **SELL** (negative total return of > -20% over the next 3 months, with identifiable share price catalysts within this time frame)
- Footnote in italics: *\*Share price appreciation + dividends*
- Below box: "Sources for all charts and tables are DBS unless otherwise specified."

---

## 6. PAGES 3-7 -- DISCLAIMER & LEGAL BOILERPLATE

### 6.1 Section Headers (Bold, Underlined)
These sections appear in this exact order:

1. **GENERAL DISCLOSURE/DISCLAIMER**
   - Opens with bold: "This report is prepared by DBS Bank Ltd." (or "DBS Bank (Hong Kong) Limited" depending on preparing entity)
   - Multiple paragraphs of standard legal text

2. **ANALYST CERTIFICATION**
   - Standard certification language

3. **COMPANY-SPECIFIC / REGULATORY DISCLOSURES**
   - Numbered items:
     1. Proprietary position disclosure (with company name and date)
   - **Compensation for investment banking services:** (bold, underlined sub-header)
     2. DBSVUSA disclosure
   - **Disclosure of previous investment recommendation produced:** (bold, underlined sub-header)
     3. Previous recommendations disclosure

4. **RESTRICTIONS ON DISTRIBUTION**
   - Two-column format: Country/region name (bold) on left, description text on right
   - Countries in order: General, Australia, Hong Kong, Indonesia, Malaysia, Singapore, Thailand, United Kingdom, Dubai International Financial Centre, United States, Other jurisdictions

5. **DBS REGIONAL RESEARCH OFFICES**
   - Two-column layout:
     - Left: HONG KONG (with full address), INDONESIA (with full address)
     - Right: SINGAPORE (with full address), THAILAND (with full address)
   - Office names in bold, company names in bold

---

## 7. COLOR REFERENCE

| Element | Color | Hex |
|---------|-------|-----|
| Body text | Black | #000000 |
| Page background | White | #FFFFFF |
| DBS logo star | Red | #ED1C24 |
| Title banner | Black | #000000 |
| Title banner text | White | #FFFFFF |
| Company stock line (indexed chart) | Orange/Gold | #D4A843 |
| S&P 500 line | Dark gray | #333333 |
| Share price line (TP chart) | Red/Coral | #CC3333 |
| Target price line (TP chart) | Black | #000000 |
| TP history table header | Red/Coral | #CC3333 |
| Table rules | Light gray | #CCCCCC |
| Chart gridlines | Light gray | #E0E0E0 |
| Section header underlines | Light gray | #CCCCCC |

---

## 8. FONT SIZES REFERENCE

| Element | Size | Weight |
|---------|------|--------|
| Page header ("DBS Group Research") | 9pt | Regular |
| Banner text ("US EQUITY RESEARCH") | 18-20pt | Regular, uppercase |
| Date line | 10pt | Regular |
| Company name | 18pt | Regular |
| Subtitle/tagline | 10pt | Regular |
| Section headers | 11pt | Regular |
| Body text | 9.5pt | Regular |
| Bold lead-in sentences | 9.5pt | Bold + Underline |
| Table body | 9pt | Regular |
| Table headers | 9.5pt | Bold |
| Source attributions | 8pt | Regular/Italic |
| Disclaimer footer | 7pt | Regular |
| Footnotes | 7pt | Regular |
