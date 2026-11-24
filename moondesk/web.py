"""Connect the lunar disk desk, source-point inspection, and saved evidence."""

from io import BytesIO
from pathlib import Path

from flask import Blueprint, Response, jsonify, render_template, request, send_file

from image_lab.validation import Invalid

from . import analysis, demo, frame, model, output
from .store import Library


def payload():
    return model.document(request.get_json(silent=True))


def png(data, filename=None):
    return send_file(BytesIO(data), mimetype="image/png", as_attachment=bool(filename),
                     download_name=filename, max_age=0)


def install(app, root):
    library = Library(root)
    proof = Blueprint("proof", __name__, url_prefix="/moon")

    @app.get("/")
    def proof_desk():
        return render_template("moondesk.html")

    @proof.get("/health")
    def health():
        return jsonify(status="ready", service="moondesk")

    @proof.get("/config")
    def configuration():
        return jsonify(modes=model.MODES, lights=model.LIGHTS, tools=model.TOOLS)

    @proof.get("/panels")
    def panels():
        return jsonify(library.list())

    @proof.post("/demo")
    def practice():
        return jsonify(demo.create(library)), 201

    @proof.post("/panels")
    def upload():
        inputs = {}
        for role in ("a", "b"):
            file = request.files.get(role)
            if file and file.filename:
                image = frame.decode(file.stream.read(model.MAX_UPLOAD + 1))
                inputs[role] = image, model.text(Path(file.filename).name)
        if "a" not in inputs:
            raise Invalid("请先导入一张主照片。")
        size = inputs["a"][0].size
        if any(image.size != size for image, _ in inputs.values()):
            raise Invalid("请使用已对齐、尺寸相同的对照照片。")
        title = model.text(request.form.get("title") or Path(inputs["a"][1]).stem)
        return jsonify(library.create(title, inputs, model.defaults(*size))), 201

    @proof.get("/panels/<panel_id>")
    def panel(panel_id):
        return jsonify(library.panel(panel_id))

    @proof.get("/panels/<panel_id>/source/<role>.png")
    def source(panel_id, role):
        archive = request.args.get("report")
        record = library.report(panel_id, archive) if archive else library.panel(panel_id)
        path = library.source(record, role)
        return send_file(path, mimetype="image/png", as_attachment="download" in request.args,
                         download_name=f"moon-{role}.png", max_age=0)

    @proof.post("/panels/<panel_id>/prepared")
    def preparation(panel_id):
        chosen = payload()
        project = model.text(chosen.get("project"))
        result = model.text(chosen.get("result"))
        preparation = app.extensions["results"]
        meta = preparation.metadata(project, result)
        if meta["stale"]:
            raise Invalid("请更新照片准备工程后再接入。")
        panel = library.panel(panel_id)
        if (meta["width"], meta["height"]) != (panel["width"], panel["height"]):
            raise Invalid("准备结果须与主照片尺寸相同。")
        source_path = preparation.image_path(project, "image", result)
        photograph = frame.decode(source_path.read_bytes())
        provenance = {"project": project, "result": result}
        return jsonify(library.set_companion(panel_id, chosen.get("revision"), photograph,
                                                model.text(meta["name"]), provenance))

    @proof.put("/panels/<panel_id>/settings")
    def update_settings(panel_id):
        chosen = payload()
        panel = library.panel(panel_id)
        settings = model.settings(chosen.get("settings"), panel)
        return jsonify(library.revise(panel_id, chosen.get("revision"), settings))

    @proof.post("/panels/<panel_id>/preview")
    def preview(panel_id):
        chosen = payload()
        panel = library.panel(panel_id)
        settings = model.settings(chosen.get("settings"), panel)
        return jsonify(analysis.report(library, panel, settings, model.mode(chosen.get("mode"))))

    @proof.post("/panels/<panel_id>/preview.png")
    def preview_overlay(panel_id):
        chosen = payload()
        panel = library.panel(panel_id)
        settings = model.settings(chosen.get("settings"), panel)
        report = analysis.report(library, panel, settings, model.mode(chosen.get("mode")))
        return png(output.overlay(library, panel, report))

    @proof.post("/panels/<panel_id>/point")
    def take_reading(panel_id):
        chosen = payload()
        archive_id = chosen.get("report")
        if archive_id:
            record = library.report(panel_id, archive_id)
            settings = record["settings"]
        else:
            record = library.panel(panel_id)
            settings = model.settings(chosen.get("settings"), record)
        point = model.point(chosen.get("point"), record)
        return jsonify(analysis.point_reading(library, record, settings, point))

    @proof.post("/panels/<panel_id>/reports")
    def archive(panel_id):
        chosen = payload()
        panel = library.panel(panel_id)
        revision = chosen.get("revision")
        library.require_revision(panel, revision)
        settings = model.settings(chosen.get("settings"), panel)
        report = analysis.report(library, panel, settings, model.mode(chosen.get("mode")))
        current, saved = library.retain(panel_id, revision, report)
        return jsonify(panel=current, report=saved), 201

    @proof.get("/panels/<panel_id>/reports/<report_id>")
    def retrieve(panel_id, report_id):
        return jsonify(library.report(panel_id, report_id))

    @proof.get("/panels/<panel_id>/reports/<report_id>/report.csv")
    def saved_csv(panel_id, report_id):
        report = library.report(panel_id, report_id)
        headers = {"Content-Disposition": f'attachment; filename="moon-{report["mode"]}.csv"'}
        return Response(output.csv_text(report), mimetype="text/csv", headers=headers)

    @proof.get("/panels/<panel_id>/reports/<report_id>/overlay.png")
    def saved_overlay(panel_id, report_id):
        report = library.report(panel_id, report_id)
        return png(output.overlay(library, report, report), "moon-inspection.png" if "download" in request.args else None)

    app.register_blueprint(proof)
