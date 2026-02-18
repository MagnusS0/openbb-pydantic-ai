<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Research Report - Style Template</title>
<style>
  /* ============================================================
     RESEARCH REPORT -- GENERIC STYLE TEMPLATE
     Reusable style template for PDF-ready output
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

  /* --- Recurring: Company Logo/Branding --- */
  .company-logo {
    text-align: right;
    margin-top: 8px;
    font-size: 18pt;
    font-weight: bold;
  }
  .company-logo .accent { color: #CC3333; }
  .company-logo .text { color: #000000; }

  /* ============================================================
     PAGE 1: COVER PAGE STYLES
     ============================================================ */

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

  .report-date {
    text-align: right;
    font-size: 10pt;
    margin-bottom: 10px;
  }

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

  .company-name {
    font-size: 18pt;
    font-weight: normal;
    margin-bottom: 4px;
  }

  .subtitle {
    font-size: 10pt;
    font-weight: normal;
    margin-bottom: 14px;
    color: #000000;
  }

  .analyst-block {
    font-size: 9pt;
    margin-bottom: 14px;
  }
  .analyst-block .label {
    color: #666666;
  }

  .section-header {
    font-size: 11pt;
    font-weight: normal;
    border-bottom: 1px solid #CCCCCC;
    padding-bottom: 2px;
    margin-top: 12px;
    margin-bottom: 8px;
  }

  .body-text {
    text-align: justify;
    font-size: 9.5pt;
    line-height: 1.35;
    margin-bottom: 6px;
  }

  .body-text .lead-in {
    font-weight: bold;
    text-decoration: underline;
  }

  /* ============================================================
     KEY FINANCIAL DATA TABLE STYLES
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
     FINANCIAL TABLES STYLES
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

  .fin-table .sub-row td:first-child {
    padding-left: 15px;
    font-style: italic;
  }
  .fin-table .sub-row td {
    font-style: italic;
  }

  .fin-table .gap-row td {
    border-bottom: none;
    padding: 4px 0;
  }

  .fin-source {
    font-size: 8pt;
    margin-bottom: 10px;
  }

  /* ============================================================
     TARGET PRICE & RATINGS HISTORY STYLES
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
     DISCLAIMER & BOILERPLATE STYLES
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

<!-- PAGE STRUCTURE EXAMPLES -->
<div class="page">
  <div class="page-header">[Company] Research Report</div>
  <div class="title-banner">PAGE TITLE</div>

  <!-- Two-column layout example (e.g., cover page) -->
  <div class="cover-layout">
    <div class="cover-left">
      <div class="company-name">Company Name</div>
      <div class="subtitle">Subtitle or tagline</div>
      <div class="section-header">Section Header</div>
      <p class="body-text">Body paragraph text goes here.</p>
      <p class="body-text"><span class="lead-in">Bold Lead-In:</span> Following text content.</p>
    </div>

    <div class="cover-right">
      <div class="analyst-block">
        <div class="label">Analyst:</div>
        <div>Name and email</div>
      </div>

      <div class="kfd-title">Key Data Table</div>
      <table class="kfd-table">
        <tr><td>Label</td><td>Value</td></tr>
        <tr><td>Label</td><td>Value</td></tr>
      </table>
      <div class="kfd-source">Source: Data Source</div>

      <div class="chart-title">Chart Title</div>
      <div class="chart-container">
        <!-- SVG chart content -->
      </div>
      <div class="chart-source">Source: Data Source</div>
    </div>
  </div>

  <div class="page-footer">Footer text goes here</div>
  <div class="company-logo"><span class="accent">●</span><span class="text">[LOGO]</span></div>
</div>

<!-- MULTI-PAGE EXAMPLE -->
<div class="page page-break">
  <div class="page-header">[Company] Research Report</div>

  <div class="fin-section-title">Financial Table Section</div>
  <table class="fin-table">
    <thead>
      <tr><th>Label</th><th>Value 1</th><th>Value 2</th></tr>
    </thead>
    <tbody>
      <tr><td>Row Label</td><td>Value</td><td>Value</td></tr>
      <tr class="sub-row"><td>Sub-row Label</td><td>Value</td><td>Value</td></tr>
      <tr class="gap-row"><td colspan="3"></td></tr>
      <tr><td>Another Row</td><td>Value</td><td>Value</td></tr>
    </tbody>
  </table>
  <div class="fin-source">Source: Data Source</div>

  <div class="page-footer">Footer text goes here</div>
  <div class="company-logo"><span class="accent">●</span><span class="text">[LOGO]</span></div>
</div>

<!-- RATINGS/HISTORY EXAMPLE -->
<div class="page page-break">
  <div class="page-header">[Company] Research Report</div>

  <div class="tp-section-title">Ratings & History</div>
  <div class="tp-layout">
    <div class="tp-chart">
      <div class="chart-container">
        <!-- SVG chart -->
      </div>
    </div>
    <div class="tp-table-wrap">
      <table class="tp-history-table">
        <thead>
          <tr><th>Date</th><th>Price</th><th>Rating</th></tr>
        </thead>
        <tbody>
          <tr><td>01/01/25</td><td>Value</td><td>BUY</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <div class="rating-box">
    Rating definitions and key metadata here.
  </div>

  <div class="page-footer">Footer text goes here</div>
  <div class="company-logo"><span class="accent">●</span><span class="text">[LOGO]</span></div>
</div>

<!-- DISCLAIMER/BOILERPLATE EXAMPLE -->
<div class="page page-break">
  <div class="page-header">[Company] Research Report</div>

  <div class="disclaimer-section-header">DISCLAIMER SECTION</div>
  <p class="disclaimer-text">Disclaimer text content.</p>

  <div class="disclaimer-sub-header">SUB-SECTION HEADER</div>
  <p class="disclaimer-text">Sub-section content.</p>

  <div class="offices-title">REGIONAL OFFICES</div>
  <div class="offices-grid">
    <div class="office">
      <div class="office-region">REGION</div>
      <div class="office-name">Office Name</div>
      <div>Contact information</div>
    </div>
    <div class="office">
      <div class="office-region">REGION</div>
      <div class="office-name">Office Name</div>
      <div>Contact information</div>
    </div>
  </div>

  <div class="page-footer">Footer text goes here</div>
  <div class="company-logo"><span class="accent">●</span><span class="text">[LOGO]</span></div>
</div>

</body>
</html>
