// A live frame-rate readout: fps, p95 and max of the frame interval, in the GUI and an overlay.
//
// A live frame-rate readout: the GUI's Performance folder and a small corner overlay (F) show how
// evenly the loop renders, from the animation-frame timestamps of the frames it renders (the
// vblank times the browser hands to rAF, which the frame pacer steps by), so they read the cadence
// the page asks the display for, not the time the tick takes.
//
// - fps: rendered frames over the last second.
// - p95 and max: the frame interval's 95th percentile and maximum over the last few seconds.
// - late: intervals longer than the paced interval by a refresh (more than half a refresh over it,
//   as the timestamps jitter a little), and hitches: intervals over 25 ms, both counted since the
//   page was revealed or since the last reset.
//
// Recording a frame is a store into a ring and two comparisons; the readout is rebuilt a few times
// a second and written to the page only while it is shown. Nothing is drawn into the canvas, so
// captures are unchanged.

// seconds the p95 and max span, seconds between readouts, the interval (ms) counted as a hitch
const SPAN = 5, REFRESH = 0.25, HITCH = 25;
// ring of frame intervals: SPAN seconds at up to 360 Hz
const CAPACITY = 2048;
const OVERLAY_KEY = 'KeyF';

export interface FrameReadout {
	fps: string;
	p95: string;
	max: string;
	late: string;
	hitches: string;
}

export class FrameStats {

	/** The text lines (GUI controllers bind to these). */
	readonly readout: FrameReadout = { fps: '–', p95: '–', max: '–', late: '0', hitches: '0' };
	/** Rendered frames, late frames and hitches since the last reset. */
	frames = 0;
	late = 0;
	hitches = 0;
	/** Called after each readout (a few times a second) while `visible()` says it is shown. */
	onReadout: ( () => void ) | null = null;
	/** Whether a readout is on screen (the GUI's folder or the overlay); otherwise only counted. */
	visible: () => boolean = () => false;
	/** Extra lines for the overlay (render scale, GPU time), read at each readout while it shows. */
	extra: () => string = () => '';
	private readonly times = new Float64Array( CAPACITY );
	private readonly intervals = new Float64Array( CAPACITY );
	private readonly sorted = new Float64Array( CAPACITY );
	private head = 0;
	private count = 0;
	private previous = NaN;
	private lastReadout = - Infinity;
	private overlay: HTMLElement | null = null;

	private readonly host: HTMLElement;

	/** `host`: the element the overlay is drawn in (the page's body), whose document and window
	 * the readout listens to. */
	constructor( host: HTMLElement ) {

		this.host = host;
		// a hidden page stops its loop: the interval across the gap is no frame of the display's
		host.ownerDocument.addEventListener( 'visibilitychange', () => { this.previous = NaN; } );
		host.ownerDocument.defaultView?.addEventListener( 'keydown', ( e ) => {

			if ( e.code !== OVERLAY_KEY || e.repeat || e.ctrlKey || e.metaKey || e.altKey ) return;
			const target = e.target as HTMLElement | null;
			if ( target && ( target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable ) ) return;
			this.toggleOverlay();

		} );

	}

	/** One rendered frame at `time` (its rAF timestamp, ms); `interval` is the paced frame interval
	 * and `period` the display's refresh (ms). */
	record( time: number, interval: number, period: number ): void {

		const delta = time - this.previous;
		this.previous = time;
		if ( delta > 0 ) {

			const i = this.head;
			this.times[ i ] = time;
			this.intervals[ i ] = delta;
			this.head = ( i + 1 ) % CAPACITY;
			if ( this.count < CAPACITY ) this.count ++;
			this.frames ++;
			if ( delta > interval + period * 0.5 ) this.late ++;
			if ( delta > HITCH ) this.hitches ++;

		}
		if ( time - this.lastReadout >= REFRESH * 1000 ) {

			this.lastReadout = time;
			if ( this.visible() || this.overlay !== null ) this.update( time );

		}

	}

	/** Clears the late and hitch counts (and the window). */
	reset(): void {

		this.frames = this.late = this.hitches = 0;
		this.count = 0;
		this.previous = NaN;
		this.lastReadout = - Infinity;

	}

	toggleOverlay(): void {

		if ( this.overlay ) {

			this.overlay.remove();
			this.overlay = null;
			return;

		}
		const el = this.host.ownerDocument.createElement( 'div' );
		// plain DOM over the canvas, no backdrop filter (one over the WebGPU canvas cost frames)
		el.style.cssText = 'position:fixed;left:8px;top:8px;z-index:20;pointer-events:none;padding:5px 8px;border-radius:4px;' +
			'background:rgba(8,20,24,0.72);color:#d8eaec;font:11px/1.35 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre';
		this.host.appendChild( el );
		this.overlay = el;
		this.lastReadout = - Infinity;

	}

	private update( now: number ): void {

		const r = this.readout;
		// walk back from the newest frame: the last second for the rate, SPAN seconds for the spread
		let inSecond = 0, n = 0, max = 0;
		for ( let k = 0; k < this.count; k ++ ) {

			const i = ( this.head - 1 - k + CAPACITY ) % CAPACITY;
			const age = now - this.times[ i ];
			if ( age > SPAN * 1000 ) break;
			if ( age < 1000 ) inSecond ++;
			const delta = this.intervals[ i ];
			this.sorted[ n ++ ] = delta;
			if ( delta > max ) max = delta;

		}
		const spread = this.sorted.subarray( 0, n ).sort();
		r.fps = n > 0 ? inSecond.toFixed( 0 ) : '–';
		r.p95 = n > 0 ? `${ spread[ Math.min( n - 1, Math.floor( n * 0.95 ) ) ].toFixed( 1 ) } ms` : '–';
		r.max = n > 0 ? `${ max.toFixed( 1 ) } ms` : '–';
		const share = this.frames > 0 ? this.late / this.frames * 100 : 0;
		r.late = `${ this.late } (${ share.toFixed( share < 10 ? 2 : 1 ) } %)`;
		r.hitches = `${ this.hitches }`;
		if ( this.overlay ) {

			this.overlay.textContent = `${ r.fps } fps   p95 ${ r.p95 }   max ${ r.max }\n` +
				`late ${ r.late }   >25 ms ${ r.hitches } / ${ this.frames }\n${ this.extra() }`;

		}
		if ( this.visible() ) this.onReadout?.();

	}

}
