/*jshint browser: true */
/*jshint -W097 */
/*jshint -W117 */
/*jshint -W061 */
"use strict";

/**
 *      dependencies:
 *          ENGINE
 *          GenericTimers
 */

const SUBTITLE = {
    VERSION: "1.00",
    CSS: "color: #7A0",
    layer: "subtitle",
    fs: 18,
    font: "Times",
    x: null,
    y: null,
    color: "FFF",
    init(layerString = "subtitle", fs = 18, font = "Times", align = "center") {
        this.setLayer(layerString);
        this.fs = fs;
        this.font = font;
        let CTX = LAYER[this.layer];
        CTX.font = `${this.fs}px ${this.font}`;
        CTX.shadowColor = "#111";
        CTX.shadowOffsetX = 1;
        CTX.shadowOffsetY = 1;
        CTX.shadowBlur = 0;
        CTX.textAlign = align;
        this.x = CTX.canvas.width / 2;
        this.y = CTX.canvas.height - this.fs;
    },
    setLayer(layerString) {
        this.layer = layerString;
    },
    getCTX() {
        return LAYER[this[layerString]];
    },
    cache: null,
    subtitle(text, color = "#FFF") {
        this.cache = null;
        ENGINE.clearLayer(this.layer);
        const CTX = LAYER[this.layer];
        this.color = color;
        this.subtitleWrite(CTX, text);
    },
    subtitleAdd(text) {
        const CTX = LAYER[this.layer];
        text = this.cache + text;
        this.subtitleWrite(CTX, text);
    },
    subtitleWrite(CTX, text) {
        CTX.fillStyle = this.color;
        CTX.fillText(text, this.x, this.y);
    },
    timedSubtitle(text, color, timer) {
        this.subtitle(text, color);
        GenericTimers.subTimer(timer);
    }

};

//END
console.log(`%cSUBTITLE ${SUBTITLE.VERSION} loaded.`, SUBTITLE.CSS);