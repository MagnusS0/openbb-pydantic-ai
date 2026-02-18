<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DBS Group Research - US Equity Research - {{COMPANY_NAME}}</title>
<style>
  /* ============================================================
     DBS GROUP RESEARCH -- US EQUITY RESEARCH REPORT
     Self-contained HTML template for PDF-ready output
     ============================================================ */

  /* --- Page Setup --- */
  @page {
    size: A4 portrait;
    margin: 15mm 15mm 25mm 20mm;
  }

  @media print {
    body { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
    .page-break { page-break-before: always; }
    .no-break { page-break-inside: avoid; }
  }

  /* --- Reset & Base --- */
  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: Calibri, 'Segoe UI', Arial, Helvetica, sans-serif;
    font-size: 9.5pt;
    line-height: 1.35;
    color: #000000;
    background: #FFFFFF;
  }

  /* --- Page Container --- */
  .page {
    width: 170mm; /* A4 minus margins */
    margin: 0 auto;
    position: relative;
    padding-bottom: 20mm;
  }

  /* --- Recurring: Top-Right Header --- */
  .page-header {
    text-align: right;
    font-size: 9pt;
    color: #000000;
    margin-bottom: 8px;
    font-weight: normal;
  }

  /* --- Recurring: Bottom Disclaimer Footer --- */
  .page-footer {
    font-size: 7pt;
    color: #000000;
    line-height: 1.3;
    border-top: 0.5px solid #CC3333;
    padding-top: 4px;
    margin-top: auto;
    position: relative;
  }

  /* --- Recurring: DBS Logo --- */
  .dbs-logo {
    text-align: right;
    margin-top: 8px;
    font-size: 18pt;
    font-weight: bold;
  }
  .dbs-logo .star { color: #ED1C24; }
  .dbs-logo .text { color: #000000; }

  /* ============================================================
     PAGE 1: COVER PAGE
     ============================================================ */

  /* --- Title Banner --- */
  .title-banner {
    background-color: #000000;
    color: #FFFFFF;
    padding: 6px 12px;
    font-size: 18pt;
    font-family: Arial, Helvetica, sans-serif;
    font-weight: normal;
    letter-spacing: 0.5px;
    width: 100%;
    margin-bottom: 4px;
  }

  /* --- Date --- */
  .report-date {
    text-align: right;
    font-size: 10pt;
    margin-bottom: 10px;
  }

  /* --- Two-Column Cover Layout --- */
  .cover-layout {
    display: flex;
    gap: 20px;
  }
  .cover-left {
    flex: 0 0 58%;
    max-width: 58%;
  }
  .cover-right {
    flex: 0 0 38%;
    max-width: 38%;
  }

  /* --- Company Name --- */
  .company-name {
    font-size: 18pt;
    font-weight: normal;
    margin-bottom: 4px;
  }

  /* --- Subtitle --- */
  .subtitle {
    font-size: 10pt;
    font-weight: normal;
    margin-bottom: 14px;
    color: #000000;
  }

  /* --- Analyst Block --- */
  .analyst-block {
    font-size: 9pt;
    margin-bottom: 14px;
  }
  .analyst-block .label {
    color: #666666;
  }

  /* --- Section Headers --- */
  .section-header {
    font-size: 11pt;
    font-weight: normal;
    border-bottom: 1px solid #CCCCCC;
    padding-bottom: 2px;
    margin-top: 12px;
    margin-bottom: 8px;
  }

  /* --- Body Text --- */
  .body-text {
    text-align: justify;
    font-size: 9.5pt;
    line-height: 1.35;
    margin-bottom: 6px;
  }

  /* --- Bold Lead-In (bold + underline per reference PDFs) --- */
  .body-text .lead-in {
    font-weight: bold;
    text-decoration: underline;
  }

  /* ============================================================
     KEY FINANCIAL DATA TABLE (Right Column)
     ============================================================ */

  .kfd-title {
    font-size: 11pt;
    font-weight: normal;
    margin-bottom: 6px;
  }

  .kfd-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 9pt;
  }
  .kfd-table td {
    padding: 2px 0;
    border-bottom: 0.5px solid #E0E0E0;
  }
  .kfd-table td:first-child {
    text-align: left;
  }
  .kfd-table td:last-child {
    text-align: right;
    font-weight: normal;
  }
  .kfd-source {
    font-size: 8pt;
    font-style: italic;
    margin-top: 2px;
    margin-bottom: 10px;
  }
  .kfd-closing {
    font-size: 8pt;
    font-style: italic;
    margin-bottom: 2px;
  }

  /* --- Chart Title --- */
  .chart-title {
    font-size: 9pt;
    font-weight: normal;
    margin-bottom: 4px;
  }
  .chart-source {
    font-size: 8pt;
    margin-top: 2px;
  }

  /* ============================================================
     PAGE 2: FINANCIAL TABLES
     ============================================================ */

  .fin-section-title {
    font-size: 10pt;
    font-weight: normal;
    margin-bottom: 4px;
    margin-top: 14px;
  }

  .fin-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 9pt;
    margin-bottom: 4px;
  }
  .fin-table thead th {
    font-weight: bold;
    text-align: right;
    padding: 3px 6px;
    border-bottom: 1px solid #000000;
    font-size: 9pt;
  }
  .fin-table thead th:first-child {
    text-align: left;
  }
  .fin-table tbody td {
    text-align: right;
    padding: 2px 6px;
    border-bottom: 0.5px solid #E0E0E0;
  }
  .fin-table tbody td:first-child {
    text-align: left;
  }

  /* Sub-row styling (% y/y rows) */
  .fin-table .sub-row td:first-child {
    padding-left: 15px;
    font-style: italic;
  }
  .fin-table .sub-row td {
    font-style: italic;
  }

  /* Gap row */
  .fin-table .gap-row td {
    border-bottom: none;
    padding: 4px 0;
  }

  .fin-source {
    font-size: 8pt;
    margin-bottom: 10px;
  }

  /* ============================================================
     PAGE 3: TARGET PRICE & RATINGS HISTORY
     ============================================================ */

  .tp-section-title {
    font-size: 11pt;
    font-weight: bold;
    text-decoration: underline;
    margin-bottom: 8px;
  }

  .tp-layout {
    display: flex;
    gap: 15px;
    margin-bottom: 10px;
  }
  .tp-chart { flex: 0 0 55%; }
  .tp-table-wrap { flex: 0 0 42%; }

  .tp-history-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 8pt;
  }
  .tp-history-table thead th {
    background-color: #CC3333;
    color: #FFFFFF;
    padding: 3px 4px;
    text-align: center;
    font-weight: bold;
    font-size: 7.5pt;
  }
  .tp-history-table tbody td {
    padding: 3px 4px;
    text-align: center;
    border-bottom: 0.5px solid #E0E0E0;
    font-size: 8pt;
  }

  .tp-attribution {
    font-size: 8pt;
    margin-top: 6px;
    margin-bottom: 12px;
  }

  /* --- Rating Definitions Box --- */
  .rating-box {
    border: 1px solid #000000;
    padding: 8px 10px;
    font-size: 8.5pt;
    line-height: 1.4;
    margin-bottom: 10px;
  }
  .rating-box .rating-name {
    font-weight: bold;
    text-decoration: underline;
  }
  .rating-box .footnote {
    font-style: italic;
    margin-top: 4px;
  }

  /* ============================================================
     PAGES 4-7: DISCLAIMER BOILERPLATE
     ============================================================ */

  .disclaimer-section-header {
    font-size: 10pt;
    font-weight: bold;
    text-decoration: underline;
    margin-top: 14px;
    margin-bottom: 6px;
  }

  .disclaimer-sub-header {
    font-size: 9.5pt;
    font-weight: bold;
    text-decoration: underline;
    margin-top: 10px;
    margin-bottom: 4px;
  }

  .disclaimer-text {
    font-size: 9pt;
    line-height: 1.4;
    text-align: justify;
    margin-bottom: 8px;
  }

  /* Restrictions table (country | description) */
  .restrictions-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 9pt;
    margin-bottom: 8px;
  }
  .restrictions-table td {
    padding: 6px 8px;
    vertical-align: top;
  }
  .restrictions-table td:first-child {
    width: 15%;
    font-weight: bold;
    text-decoration: underline;
  }

  /* Regional offices */
  .offices-title {
    font-size: 10pt;
    font-weight: bold;
    border-bottom: 1px solid #000000;
    padding-bottom: 2px;
    margin-top: 16px;
    margin-bottom: 10px;
    letter-spacing: 0.5px;
  }
  .offices-grid {
    display: flex;
    gap: 30px;
    font-size: 9pt;
  }
  .offices-grid .office {
    flex: 1;
  }
  .offices-grid .office-region {
    font-weight: bold;
    margin-top: 10px;
  }
  .offices-grid .office-name {
    font-weight: bold;
  }

  /* Footnotes */
  .footnote-section {
    border-top: 1px solid #000000;
    margin-top: 20px;
    padding-top: 6px;
    font-size: 7pt;
    line-height: 1.3;
  }

  /* ============================================================
     SVG CHART DEFAULTS
     ============================================================ */
  .chart-container svg {
    width: 100%;
    height: auto;
  }
  .chart-container svg text {
    font-family: Calibri, 'Segoe UI', Arial, Helvetica, sans-serif;
  }

</style>
</head>
<body>

<!-- ============================================================
     PAGE 1: COVER PAGE
     ============================================================ -->
<div class="page" id="page1">
  <div class="page-header">DBS Group Research</div>

  <div class="title-banner">US EQUITY RESEARCH</div>
  <div class="report-date">{{REPORT_DATE}}</div>

  <div class="cover-layout">

    <!-- LEFT COLUMN: Narrative -->
    <div class="cover-left">
      <div class="company-name">{{COMPANY_NAME}}</div>
      <div class="subtitle">{{SUBTITLE_TAGLINE}}</div>

      <div class="section-header">Company Overview</div>
      <p class="body-text">{{COMPANY_OVERVIEW}}</p>

      <div class="section-header">Investment Overview</div>
      <!-- Repeat this pattern for each investment argument paragraph -->
      <p class="body-text">
        <span class="lead-in">{{ARGUMENT_1_BOLD_LEAD_IN}}</span> {{ARGUMENT_1_BODY}}
      </p>
      <p class="body-text">
        <span class="lead-in">{{ARGUMENT_2_BOLD_LEAD_IN}}</span> {{ARGUMENT_2_BODY}}
      </p>
      <p class="body-text">
        <span class="lead-in">{{ARGUMENT_3_BOLD_LEAD_IN}}</span> {{ARGUMENT_3_BODY}}
      </p>
      <!-- Final paragraph is always the rating/valuation call -->
      <p class="body-text">
        <span class="lead-in">{{RATING_CALL_BOLD}}</span> {{RATING_CALL_BODY}}
      </p>

      <div class="section-header">Risks</div>
      <p class="body-text">
        <span class="lead-in">{{RISK_BOLD_LEAD_IN}}</span> {{RISK_BODY}}
      </p>
    </div>

    <!-- RIGHT COLUMN: Data Sidebar -->
    <div class="cover-right">
      <div class="analyst-block">
        <div class="label">{{ANALYST_LABEL}}</div>
        <div>{{ANALYST_NAME_EMAIL}}</div>
      </div>

      <div class="kfd-title">Key Financial Data</div>
      <table class="kfd-table">
        <tr><td>Bloomberg Ticker</td><td>{{TICKER}} US</td></tr>
        <tr><td>Sector</td><td>{{SECTOR}}</td></tr>
        <tr><td>Share Price (USD)</td><td>{{SHARE_PRICE}}</td></tr>
        <tr><td>DBS Rating</td><td>{{DBS_RATING}}</td></tr>
        <tr><td>12-mth Target Price (USD)</td><td>{{TARGET_PRICE}}</td></tr>
        <tr><td>Market Cap (USDbn)</td><td>{{MARKET_CAP}}</td></tr>
        <tr><td>Volume (mn shares)</td><td>{{VOLUME}}</td></tr>
        <tr><td>Free float (%)</td><td>{{FREE_FLOAT}}</td></tr>
        <tr><td>Dividend yield (%)</td><td>{{DIVIDEND_YIELD}}</td></tr>
        <tr><td>Net Debt to Equity (%)</td><td>{{NET_DEBT_EQUITY}}</td></tr>
        <tr><td>Fwd. P/E (x)</td><td>{{FWD_PE}}</td></tr>
        <tr><td>P/Book (x)</td><td>{{P_BOOK}}</td></tr>
        <tr><td>ROE (%)</td><td>{{ROE}}</td></tr>
      </table>
      <div class="kfd-closing"><em>Closing Price as of {{CLOSING_DATE}}</em></div>
      <div class="kfd-source">Source: Bloomberg, DBS</div>

      <div class="chart-title">Indexed Share Price vs Composite Index Performance</div>
      <div class="chart-container">
        <!-- INSERT INDEXED PERFORMANCE SVG CHART HERE -->
        {{INDEXED_CHART_SVG}}
      </div>
      <div class="chart-source">Source: Bloomberg</div>
    </div>

  </div>

  <div class="page-footer">
    Disclaimer: The information contained in this document is intended only for use by the person to whom it has been delivered and should not be disseminated
    or distributed to third parties without our prior written consent. DBS accepts no liability whatsoever with respect to the use of this document or its contents.
    Please refer to Disclaimer found at the end of this document
  </div>
  <div class="dbs-logo"><span class="star">&#10070;</span><span class="text">DBS</span></div>
</div>


<!-- ============================================================
     PAGE 2: FINANCIAL TABLES
     ============================================================ -->
<div class="page page-break" id="page2">
  <div class="page-header">DBS Group Research</div>

  <div class="fin-section-title">Financial Summary (USDmn)</div>
  <table class="fin-table">
    <thead>
      <tr>
        <th>FY Dec</th>
        <th>FY2022A</th>
        <th>FY2023A</th>
        <th>FY2024A</th>
        <th>FY2025F</th>
        <th>FY2026F</th>
      </tr>
    </thead>
    <tbody>
      <!-- {{FINANCIAL_SUMMARY_ROWS}} -->
      <!-- Example row structure: -->
      <tr><td>Sales</td><td>{{SALES_2022}}</td><td>{{SALES_2023}}</td><td>{{SALES_2024}}</td><td>{{SALES_2025}}</td><td>{{SALES_2026}}</td></tr>
      <tr class="sub-row"><td>% y/y</td><td>{{SALES_YOY_2022}}</td><td>{{SALES_YOY_2023}}</td><td>{{SALES_YOY_2024}}</td><td>{{SALES_YOY_2025}}</td><td>{{SALES_YOY_2026}}</td></tr>
      <tr><td>Gross Profit</td><td>{{GP_2022}}</td><td>{{GP_2023}}</td><td>{{GP_2024}}</td><td>{{GP_2025}}</td><td>{{GP_2026}}</td></tr>
      <tr class="sub-row"><td>% y/y</td><td>{{GP_YOY_2022}}</td><td>{{GP_YOY_2023}}</td><td>{{GP_YOY_2024}}</td><td>{{GP_YOY_2025}}</td><td>{{GP_YOY_2026}}</td></tr>
      <tr><td>EBITDA</td><td>{{EBITDA_2022}}</td><td>{{EBITDA_2023}}</td><td>{{EBITDA_2024}}</td><td>{{EBITDA_2025}}</td><td>{{EBITDA_2026}}</td></tr>
      <tr class="sub-row"><td>% y/y</td><td>{{EBITDA_YOY_2022}}</td><td>{{EBITDA_YOY_2023}}</td><td>{{EBITDA_YOY_2024}}</td><td>{{EBITDA_YOY_2025}}</td><td>{{EBITDA_YOY_2026}}</td></tr>
      <tr><td>Net Profit (Loss)</td><td>{{NP_2022}}</td><td>{{NP_2023}}</td><td>{{NP_2024}}</td><td>{{NP_2025}}</td><td>{{NP_2026}}</td></tr>
      <tr class="sub-row"><td>% y/y</td><td>{{NP_YOY_2022}}</td><td>{{NP_YOY_2023}}</td><td>{{NP_YOY_2024}}</td><td>{{NP_YOY_2025}}</td><td>{{NP_YOY_2026}}</td></tr>
      <tr><td>FCF</td><td>{{FCF_2022}}</td><td>{{FCF_2023}}</td><td>{{FCF_2024}}</td><td>{{FCF_2025}}</td><td>{{FCF_2026}}</td></tr>
      <tr class="sub-row"><td>% y/y</td><td>{{FCF_YOY_2022}}</td><td>{{FCF_YOY_2023}}</td><td>{{FCF_YOY_2024}}</td><td>{{FCF_YOY_2025}}</td><td>{{FCF_YOY_2026}}</td></tr>
      <tr><td>CAPEX</td><td>{{CAPEX_2022}}</td><td>{{CAPEX_2023}}</td><td>{{CAPEX_2024}}</td><td>{{CAPEX_2025}}</td><td>{{CAPEX_2026}}</td></tr>
      <tr class="sub-row"><td>% y/y</td><td>{{CAPEX_YOY_2022}}</td><td>{{CAPEX_YOY_2023}}</td><td>{{CAPEX_YOY_2024}}</td><td>{{CAPEX_YOY_2025}}</td><td>{{CAPEX_YOY_2026}}</td></tr>
      <tr class="gap-row"><td colspan="6"></td></tr>
      <tr><td>EBITDA Margin (%)</td><td>{{EBITDA_M_2022}}</td><td>{{EBITDA_M_2023}}</td><td>{{EBITDA_M_2024}}</td><td>{{EBITDA_M_2025}}</td><td>{{EBITDA_M_2026}}</td></tr>
      <tr><td>Net Margin (%)</td><td>{{NET_M_2022}}</td><td>{{NET_M_2023}}</td><td>{{NET_M_2024}}</td><td>{{NET_M_2025}}</td><td>{{NET_M_2026}}</td></tr>
      <tr><td>ROA (%)</td><td>{{ROA_2022}}</td><td>{{ROA_2023}}</td><td>{{ROA_2024}}</td><td>{{ROA_2025}}</td><td>{{ROA_2026}}</td></tr>
      <tr><td>ROE (%)</td><td>{{ROE_2022}}</td><td>{{ROE_2023}}</td><td>{{ROE_2024}}</td><td>{{ROE_2025}}</td><td>{{ROE_2026}}</td></tr>
      <tr><td>Tax Rate (%)</td><td>{{TAX_2022}}</td><td>{{TAX_2023}}</td><td>{{TAX_2024}}</td><td>{{TAX_2025}}</td><td>{{TAX_2026}}</td></tr>
    </tbody>
  </table>
  <div class="fin-source">Source: Visible Alpha</div>

  <div class="fin-section-title">Valuation Metrics</div>
  <table class="fin-table">
    <thead>
      <tr>
        <th>FY Dec</th>
        <th>FY2022A</th>
        <th>FY2023A</th>
        <th>FY2024A</th>
        <th>FY2025F</th>
        <th>FY2026F</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>P/E</td><td>{{PE_2022}}</td><td>{{PE_2023}}</td><td>{{PE_2024}}</td><td>{{PE_2025}}</td><td>{{PE_2026}}</td></tr>
      <tr><td>P/B</td><td>{{PB_2022}}</td><td>{{PB_2023}}</td><td>{{PB_2024}}</td><td>{{PB_2025}}</td><td>{{PB_2026}}</td></tr>
      <tr><td>Dividend Yield</td><td>{{DY_2022}}</td><td>{{DY_2023}}</td><td>{{DY_2024}}</td><td>{{DY_2025}}</td><td>{{DY_2026}}</td></tr>
      <tr><td>EV/EBITDA (x)</td><td>{{EVEBITDA_2022}}</td><td>{{EVEBITDA_2023}}</td><td>{{EVEBITDA_2024}}</td><td>{{EVEBITDA_2025}}</td><td>{{EVEBITDA_2026}}</td></tr>
      <tr><td>FCF Yield %</td><td>{{FCFY_2022}}</td><td>{{FCFY_2023}}</td><td>{{FCFY_2024}}</td><td>{{FCFY_2025}}</td><td>{{FCFY_2026}}</td></tr>
    </tbody>
  </table>
  <div class="fin-source">Source: Visible Alpha</div>

  <div class="fin-section-title">Credit & Cashflow Metrics</div>
  <table class="fin-table">
    <thead>
      <tr>
        <th>FY Dec</th>
        <th>FY2022A</th>
        <th>FY2023A</th>
        <th>FY2024A</th>
        <th>FY2025F</th>
        <th>FY2026F</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>Debt / Equity</td><td>{{DE_2022}}</td><td>{{DE_2023}}</td><td>{{DE_2024}}</td><td>{{DE_2025}}</td><td>{{DE_2026}}</td></tr>
      <tr><td>Net Debt / Equity</td><td>{{NDE_2022}}</td><td>{{NDE_2023}}</td><td>{{NDE_2024}}</td><td>{{NDE_2025}}</td><td>{{NDE_2026}}</td></tr>
      <tr><td>Debt / Assets</td><td>{{DA_2022}}</td><td>{{DA_2023}}</td><td>{{DA_2024}}</td><td>{{DA_2025}}</td><td>{{DA_2026}}</td></tr>
      <tr><td>Net Debt / Assets</td><td>{{NDA_2022}}</td><td>{{NDA_2023}}</td><td>{{NDA_2024}}</td><td>{{NDA_2025}}</td><td>{{NDA_2026}}</td></tr>
      <tr><td>EBITDA / Int Exp</td><td>{{EIE_2022}}</td><td>{{EIE_2023}}</td><td>{{EIE_2024}}</td><td>{{EIE_2025}}</td><td>{{EIE_2026}}</td></tr>
      <tr><td>Debt / EBITDA</td><td>{{DEBITDA_2022}}</td><td>{{DEBITDA_2023}}</td><td>{{DEBITDA_2024}}</td><td>{{DEBITDA_2025}}</td><td>{{DEBITDA_2026}}</td></tr>
      <tr><td>ST Debt / Total Debt (%)</td><td>{{STDT_2022}}</td><td>{{STDT_2023}}</td><td>{{STDT_2024}}</td><td>{{STDT_2025}}</td><td>{{STDT_2026}}</td></tr>
      <tr><td>[Cash + CFO] / ST Debt</td><td>{{CCFO_2022}}</td><td>{{CCFO_2023}}</td><td>{{CCFO_2024}}</td><td>{{CCFO_2025}}</td><td>{{CCFO_2026}}</td></tr>
      <tr><td>Receivables Days</td><td>{{RD_2022}}</td><td>{{RD_2023}}</td><td>{{RD_2024}}</td><td>{{RD_2025}}</td><td>{{RD_2026}}</td></tr>
      <tr><td>Days Payable</td><td>{{DP_2022}}</td><td>{{DP_2023}}</td><td>{{DP_2024}}</td><td>{{DP_2025}}</td><td>{{DP_2026}}</td></tr>
      <tr><td>Inventory Days</td><td>{{ID_2022}}</td><td>{{ID_2023}}</td><td>{{ID_2024}}</td><td>{{ID_2025}}</td><td>{{ID_2026}}</td></tr>
    </tbody>
  </table>
  <div class="fin-source">Source: Visible Alpha</div>

  <div class="page-footer">
    Disclaimer: The information contained in this document is intended only for use by the person to whom it has been delivered and should not be disseminated
    or distributed to third parties without our prior written consent. DBS accepts no liability whatsoever with respect to the use of this document or its contents.
    Please refer to Disclaimer found at the end of this document
  </div>
  <div class="dbs-logo"><span class="star">&#10070;</span><span class="text">DBS</span></div>
</div>


<!-- ============================================================
     PAGE 3: TARGET PRICE & RATINGS HISTORY + DISCLAIMER START
     ============================================================ -->
<div class="page page-break" id="page3">
  <div class="page-header">DBS Group Research</div>

  <div class="tp-section-title">Target Price & Ratings History</div>

  <div class="tp-layout">
    <div class="tp-chart">
      <div class="chart-container">
        <!-- INSERT TARGET PRICE HISTORY SVG CHART HERE -->
        {{TP_CHART_SVG}}
      </div>
    </div>
    <div class="tp-table-wrap">
      <table class="tp-history-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Date of Report</th>
            <th>Closing Price</th>
            <th>12-m Target Price</th>
            <th>Rating</th>
          </tr>
        </thead>
        <tbody>
          <!-- {{TP_HISTORY_ROWS}} -->
          <!-- Example: <tr><td>1</td><td>01 May'25</td><td>72.55</td><td>83.00</td><td>BUY</td></tr> -->
        </tbody>
      </table>
    </div>
  </div>

  <div class="tp-attribution">
    Source: DBS<br>
    {{ANALYST_ATTRIBUTION}}
  </div>

  <div class="rating-box">
    DBS Group Research recommendations are based on an Absolute Total Return* Rating system, defined as follows:<br>
    <span class="rating-name">STRONG BUY</span> (&gt;20% total return over the next 3 months, with identifiable share price catalysts within this time frame)<br>
    <span class="rating-name">BUY</span> (&gt;15% total return over the next 12 months for small caps, &gt;10% for large caps)<br>
    <span class="rating-name">HOLD</span> (-10% to +15% total return over the next 12 months for small caps, -10% to +10% for large caps)<br>
    <span class="rating-name">FULLY VALUED</span> (negative total return, i.e., &gt; -10% over the next 12 months)<br>
    <span class="rating-name">SELL</span> (negative total return of &gt; -20% over the next 3 months, with identifiable share price catalysts within this time frame)<br>
    <div class="footnote">*Share price appreciation + dividends</div>
  </div>

  <p class="disclaimer-text">Sources for all charts and tables are DBS unless otherwise specified.</p>

  <div class="disclaimer-section-header">GENERAL DISCLOSURE/DISCLAIMER</div>
  <p class="disclaimer-text">
    <strong>This report is prepared by DBS Bank Ltd.</strong> This report is solely intended for the clients of DBS Bank Ltd, DBS Vickers Securities (Singapore) Pte Ltd,
    its respective connected and associated corporations and affiliates only and no part of this document may be (i) copied, photocopied or duplicated
    in any form or by any means or (ii) redistributed without the prior written consent of DBS Bank Ltd.
  </p>
  <p class="disclaimer-text">
    The research set out in this report is based on information obtained from sources believed to be reliable, but we (which collectively refers to DBS
    Bank Ltd, DBS Vickers Securities (Singapore) Pte Ltd, its respective connected and associated corporations, affiliates and their respective directors,
    officers, employees and agents (collectively, the "<strong>DBS Group</strong>") have not conducted due diligence on any of the companies, verified any information
    or sources or taken into account any other factors which we may consider to be relevant or appropriate in preparing the research. Accordingly,
    we do not make any representation or warranty as to the accuracy, completeness or correctness of the research set out in this report. Opinions
    expressed are subject to change without notice. This research is prepared for general circulation. Any recommendation contained in this document
    does not have regard to the specific investment objectives, financial situation and the particular needs of any specific addressee. This document
    is for the information of addressees only and is not to be taken in substitution for the exercise of judgement by addressees, who should obtain
    separate independent legal or financial advice. The DBS Group accepts no liability whatsoever for any direct, indirect and/or consequential loss
    (including any claims for loss of profit) arising from any use of and/or reliance upon this document and/or further communication given in relation
    to this document. This document is not to be construed as an offer or a solicitation of an offer to buy or sell any securities. The DBS Group, along
    with its affiliates and/or persons associated with any of them may from time to time have interests in the securities mentioned in this document.
    The DBS Group, may have positions in, and may effect transactions in securities mentioned herein and may also perform or seek to perform
    broking, investment banking and other banking services for these companies.
  </p>
  <p class="disclaimer-text">
    Any valuations, opinions, estimates, forecasts, ratings or risk assessments herein constitutes a judgment as of the date of this report, and there
    can be no assurance that future results or events will be consistent with any such valuations, opinions, estimates, forecasts, ratings or risk
    assessments. The information in this document is subject to change without notice, its accuracy is not guaranteed, it may be incomplete or
    condensed, it may not contain all material information concerning the company (or companies) referred to in this report and the DBS Group is
    under no obligation to update the information in this report.
  </p>

  <div class="page-footer">
    Disclaimer: The information contained in this document is intended only for use by the person to whom it has been delivered and should not be disseminated
    or distributed to third parties without our prior written consent. DBS accepts no liability whatsoever with respect to the use of this document or its contents.
    Please refer to Disclaimer found at the end of this document
  </div>
  <div class="dbs-logo"><span class="star">&#10070;</span><span class="text">DBS</span></div>
</div>


<!-- ============================================================
     PAGES 4-7: Remaining disclaimer pages follow the same
     structure. When generating the report, include the full
     boilerplate text from the spec for:
     - Continuation of General Disclosure
     - Analyst Certification
     - Company-Specific / Regulatory Disclosures
     - Restrictions on Distribution (by country)
     - DBS Regional Research Offices
     ============================================================ -->

<!-- PAGE 4 -->
<div class="page page-break" id="page4">
  <div class="page-header">DBS Group Research</div>
  <!-- {{DISCLAIMER_PAGE_4_CONTENT}} -->
  <div class="page-footer">
    Disclaimer: The information contained in this document is intended only for use by the person to whom it has been delivered and should not be disseminated
    or distributed to third parties without our prior written consent. DBS accepts no liability whatsoever with respect to the use of this document or its contents.
    Please refer to Disclaimer found at the end of this document
  </div>
  <div class="dbs-logo"><span class="star">&#10070;</span><span class="text">DBS</span></div>
</div>

<!-- PAGE 5 -->
<div class="page page-break" id="page5">
  <div class="page-header">DBS Group Research</div>
  <!-- {{DISCLAIMER_PAGE_5_CONTENT}} -->
  <div class="page-footer">
    Disclaimer: The information contained in this document is intended only for use by the person to whom it has been delivered and should not be disseminated
    or distributed to third parties without our prior written consent. DBS accepts no liability whatsoever with respect to the use of this document or its contents.
    Please refer to Disclaimer found at the end of this document
  </div>
  <div class="dbs-logo"><span class="star">&#10070;</span><span class="text">DBS</span></div>
</div>

<!-- PAGE 6 -->
<div class="page page-break" id="page6">
  <div class="page-header">DBS Group Research</div>
  <!-- {{DISCLAIMER_PAGE_6_CONTENT}} -->
  <div class="page-footer">
    Disclaimer: The information contained in this document is intended only for use by the person to whom it has been delivered and should not be disseminated
    or distributed to third parties without our prior written consent. DBS accepts no liability whatsoever with respect to the use of this document or its contents.
    Please refer to Disclaimer found at the end of this document
  </div>
  <div class="dbs-logo"><span class="star">&#10070;</span><span class="text">DBS</span></div>
</div>

<!-- PAGE 7 -->
<div class="page page-break" id="page7">
  <div class="page-header">DBS Group Research</div>
  <!-- {{DISCLAIMER_PAGE_7_CONTENT}} -->

  <div class="offices-title">DBS REGIONAL RESEARCH OFFICES</div>
  <div class="offices-grid">
    <div class="office">
      <div class="office-region">HONG KONG</div>
      <div class="office-name">DBS Bank (Hong Kong) Ltd</div>
      <div>Contact: Dennis Lam</div>
      <div>13th Floor One Island East,</div>
      <div>18 Westlands Road,</div>
      <div>Quarry Bay, Hong Kong</div>
      <div>Tel: 852 3668 4181</div>
      <div>Fax: 852 2521 1812</div>
      <div>e-mail: dbsvhk@dbs.com</div>

      <div class="office-region" style="margin-top:18px;">INDONESIA</div>
      <div class="office-name">PT DBS Vickers Sekuritas (Indonesia)</div>
      <div>Contact: William Simadiputra</div>
      <div>DBS Bank Tower</div>
      <div>Ciputra World 1, 32/F</div>
      <div>Jl. Prof. Dr. Satrio Kav. 3-5</div>
      <div>Jakarta 12940, Indonesia</div>
      <div>Tel: 62 21 3003 4900</div>
      <div>Fax: 6221 3003 4943</div>
      <div>e-mail: indonesiaresearch@dbs.com</div>
    </div>
    <div class="office">
      <div class="office-region">SINGAPORE</div>
      <div class="office-name">DBS Bank Ltd</div>
      <div>Contact: Andy Sim</div>
      <div>12 Marina Boulevard,</div>
      <div>Marina Bay Financial Centre Tower 3</div>
      <div>Singapore 018982</div>
      <div>Tel: 65 6878 8888</div>
      <div>e-mail: groupresearch@dbs.com</div>
      <div>Company Regn. No. 196800306E</div>

      <div class="office-region" style="margin-top:18px;">THAILAND</div>
      <div class="office-name">DBS Vickers Securities (Thailand) Co Ltd</div>
      <div>Contact: Chanpen Sirithanarattanakul</div>
      <div>989 Siam Piwat Tower Building,</div>
      <div>9th, 14th-15th Floor</div>
      <div>Rama 1 Road, Pathumwan,</div>
      <div>Bangkok Thailand 10330</div>
      <div>Tel. 66 2 857 7831</div>
      <div>Fax: 66 2 658 1269</div>
      <div>e-mail: DBSVTresearch@dbs.com</div>
      <div>Company Regn. No 0105539127012</div>
      <div>Securities and Exchange Commission, Thailand</div>
    </div>
  </div>

  <div class="page-footer">
    Disclaimer: The information contained in this document is intended only for use by the person to whom it has been delivered and should not be disseminated
    or distributed to third parties without our prior written consent. DBS accepts no liability whatsoever with respect to the use of this document or its contents.
    Please refer to Disclaimer found at the end of this document
  </div>
  <div class="dbs-logo"><span class="star">&#10070;</span><span class="text">DBS</span></div>
</div>

</body>
</html>
