

typedef float4 point;
typedef float4 vector;
typedef float4 color;
typedef float4 sphere;


vector
Bounce( vector in, vector n )
{
	vector out = in - n*(vector)( 2.*dot(in.xyz, n.xyz) );
	out.w = 0.;
	return out;
}

vector
BounceSphere( point p, vector v, sphere s )
{
	vector n;
	n.xyz = fast_normalize( p.xyz - s.xyz );
	n.w = 0.;
	return Bounce( v, n );
}

bool
IsInsideSphere( point p, sphere s )
{
	float r = fast_length( p.xyz - s.xyz );
	return  ( r < s.w );
}

kernel
void
Particle( global point *dPobj, global vector *dVel, global color *dCobj )
{
	const float4 G       = (float4) ( 0., -9.8, 0., 0. );
	const float  DT      = 0.1;
	// const float  DT      = 0.05;						// slower speed for screenshots
	const sphere Sphere1 = (sphere)( -100., -800., 0.,  600. );
	const sphere Sphere2 = (sphere)( 500.,  80.,  0.,  200. );
	
	int gid = get_global_id( 0 );						// particle #

	// current values
	point p  = dPobj[gid];
	vector v = dVel[gid];
	color c  = dCobj[gid];
	
	// next values
	color cp  = dCobj[gid];
	point pp  = p + v*DT + G * (point)(.5*DT*DT); 	
	vector vp = v + G*DT;
	pp.w = 1.;
	vp.w = 0.;
	
	if( IsInsideSphere( pp, Sphere1) )
	{
		vp = BounceSphere( p, v, Sphere1 );
		pp = p + vp*DT + G * (point)(.5*DT*DT);
		cp = (color)( 0.000, 0.980, 0.604, 1.0 ); 		// change color to green
	}
	
	if( IsInsideSphere( pp, Sphere2) )
	{
		vp = BounceSphere( p, v, Sphere2 );
		pp = p + vp*DT + G * (point)(.5*DT*DT);
		cp = (color)( 0.000, 0.749, 1.000, 1.0 ); 		// change color to blue
	}
	
	// update values
	dPobj[gid] = pp;
	dVel[gid]  = vp;
	dCobj[gid] = cp;

}

