/*jshint browser: true */
/*jshint -W097 */
/*jshint -W117 */
/*jshint -W061 */
/*jshint -W083 */
"use strict";

/**
 *      dependencies:
 *          ENGINE
 */

const GenericTimers = {
    VERSION: "1.03",
    CSS: "color: #b785FF",
    INI: {
        INFO_TIMER_ID: "info",
        INFO_TIMER: 3,
        SUB_TIMER_ID: "subtitle",
        SUB_TIMER: 10,
    },

    clearInfo() {
        ENGINE.clearLayer("info");
    },
    clearSubtitle() {
        ENGINE.clearLayer("subtitle");
    },
    infoTimer(info_time = this.INI.INFO_TIMER) {
        let T;
        if (ENGINE.TIMERS.exists(GenericTimers.INI.INFO_TIMER_ID)) {
            T = ENGINE.TIMERS.access(GenericTimers.INI.INFO_TIMER_ID);
            T.set(info_time);
        } else {
            T = new CountDown(GenericTimers.INI.INFO_TIMER_ID, info_time, GenericTimers.clearInfo);
        }
    },
    subTimer(sub_time = this.INI.SUB_TIMER) {
        let T;
        if (ENGINE.TIMERS.exists(GenericTimers.INI.SUB_TIMER_ID)) {
            T = ENGINE.TIMERS.access(GenericTimers.INI.SUB_TIMER_ID);
            T.extend(sub_time);
        } else {
            T = new CountDown(GenericTimers.INI.SUB_TIMER_ID, sub_time, GenericTimers.clearSubtitle);
        }
    },
};


//END
console.log(`%cGenericTimers ${GenericTimers.VERSION} loaded.`, GenericTimers.CSS);