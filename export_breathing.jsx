// Export torso breathing scale keyframes (with Loop Out "pingpong") to JSON.
// 1. Select ONLY the torso layer before running this script.
// 2. Set the comp's work area to exactly one breath cycle:
//    - if your keyframes already return to the starting scale by the last
//      frame (a full self-contained breath in+out), that's a CYCLE -- set
//      LOOP_MODE below to "cycle" (the default). It plays forward once per
//      period then jumps back to the start.
//    - if you only animated HALF a breath (e.g. just the inhale) and want
//      the exhale to be its mirror image, set LOOP_MODE to "pingpong".
//    Using "pingpong" on a full cycle doubles the apparent length and can
//    make the timing feel off -- that was likely why breathing looked
//    unnatural before (the old Python code forced pingpong regardless of
//    this field).
// 3. Run this via File > Scripts > Run Script File.

var LOOP_MODE = "cycle"; // "cycle" or "pingpong" -- see note above

var comp = app.project.activeItem;
if (!comp || !(comp instanceof CompItem)) {
    alert("Open the puppy composition first.");
} else if (comp.selectedLayers.length !== 1) {
    alert("Select exactly one layer: the torso.");
} else {
    var layer = comp.selectedLayers[0];
    var scaleProp = layer.property("ADBE Transform Group").property("ADBE Scale");

    var start = comp.workAreaStart;
    var end = comp.workAreaStart + comp.workAreaDuration;
    var step = 1 / 20; // sample every 0.05 second

    var samples = [];
    for (var t = start; t < end - 1e-6; t += step) {
        var s = scaleProp.valueAtTime(t, false); // [x%, y%] (or [x,y,z])
        samples.push(Math.round(s[1] * 1000) / 1000); // vertical scale = breathing
    }

    // Build the JSON text by hand -- some AE script engines don't provide
    // the built-in JSON object, so we can't rely on JSON.stringify here.
    var text = "{";
    text += "\"fps_samples_per_second\": 20,";
    text += "\"duration_seconds\": " + (Math.round((end - start) * 1000) / 1000) + ",";
    text += "\"loop\": \"" + LOOP_MODE + "\",";
    text += "\"torso_scale_y_percent\": [" + samples.join(",") + "]";
    text += "}";

    var f = File.saveDialog("Save breathing JSON", "*.json");
    if (f) {
        f.open("w");
        f.write(text);
        f.close();
        alert("Saved " + samples.length + " samples.");
    }
}
