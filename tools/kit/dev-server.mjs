// Stops a probe at once when its URL is not the Vite dev server. Probes that import the source modules
// into the page need the dev server, which compiles them on request; a production preview does
// not serve them, and without this check such a probe waited about 90 s and then timed out.
// Called before the GPU lock is taken or a browser starts, so a wrong server costs nothing.
// Only the dev server serves /@vite/client (200); a preview or static server answers 404.

export async function requireDevServer( url, tool ) {

	let ok = false;
	try {

		const response = await fetch( new URL( '/@vite/client', url ) );
		ok = response.status === 200;

	} catch {

		ok = false;

	}

	if ( ! ok ) {

		console.error( `${ tool }: needs the dev server (npm run dev): it imports the source modules, which a production preview does not serve` );
		process.exit( 1 );

	}

}
