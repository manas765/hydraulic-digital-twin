from src.hydraulic_twin import ui_components as ui


def test_card_escapes_text_and_shows_source_badge():
    html = ui.card("<b>Label</b>", "5 < 6", "ml", "a & b", tone="ok")
    assert "<b>Label</b>" not in html
    assert "&lt;b&gt;Label&lt;/b&gt;" in html
    assert "5 &lt; 6" in html and "a &amp; b" in html
    assert "ML ESTIMATE" in html and "tone-ok" in html


def test_alerts_are_escaped_and_levelled():
    alerts = [{"level": "CRITICAL", "source": "ml_prediction",
               "message": "<script>alert(1)</script>"}]
    html = ui.alerts_html(alerts)
    assert "<script>" not in html
    assert "hx-alert CRITICAL" in html
    assert "ML PREDICTION" in html


def test_empty_alert_list_says_so():
    assert "No alerts" in ui.alerts_html([])


def test_header_and_section_are_escaped():
    assert "&amp;" in ui.header_html("A & B", "sub", "MODE")
    assert "&lt;" in ui.section_html("x < y")


def test_style_uses_theme_neutral_colours_for_panels():
    # panels must not hard-code a white background (breaks dark mode)
    card_rule = ui.STYLE.split(".hx-card {")[1].split("}")[0]
    assert "#fff" not in card_rule.lower()
    assert "rgba(" in card_rule