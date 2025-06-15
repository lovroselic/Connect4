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
    setLayer(layerString) {
        this.layer = layerString;
    },
    getCTX() {
        return LAYER[this[layerString]];
    }
};

//END
console.log(`%cSUBTITLE ${SUBTITLE.VERSION} loaded.`, SUBTITLE.CSS);