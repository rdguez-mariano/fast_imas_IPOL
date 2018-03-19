/*
 Lib match file for SURF
 
 Copyright 2013: Edouard Oyallon, Julien Rabin
 
 Version for IPOL.
 */

#include "lib_match_surf.h"

float surf_ratio=0.6;


std::vector<MatchSurf> matchDescriptor(listDescriptor * l1, listDescriptor * l2)
{
    std::vector<MatchSurf> matches;
    
    // Matching is not symmetric.
    for(int i=0;i<(int)l1->size();i++)
    {
        int position=-1;
        float d1=3;
        float d2=3;
        
        for(int j=0;j<(int)l2->size();j++)
        {
			float d=euclideanDistance((*l1)[i],(*l2)[j]);
			// We select the two closes descriptors
			if((((*l1)[i])->kP)->signLaplacian==(((*l2)[j])->kP)->signLaplacian)
			{
				d2=((d2>d)?d:d2);
				if( d1>d)
				{
					position=j;
					d2=d1;
					d1=d;
				}
			}
		}
		// Try to match it
        if(position>=0  && surf_ratio*d2>d1)
		{
            MatchSurf match;
			match.x1=((*l1)[i]->kP)->x;
            match.y1=((*l1)[i]->kP)->y;
            match.scale1=((*l1)[i]->kP)->scale;
            match.angle1=((*l1)[i]->kP)->orientation;
			match.x2=((*l2)[position]->kP)->x;
			match.y2=((*l2)[position]->kP)->y;
            match.scale2=((*l2)[position]->kP)->scale;
            match.angle2=((*l2)[position]->kP)->orientation;
			matches.push_back(match);
		}
	}
	return matches;
}



// Square of euclidean distance between two descriptors
float euclideanDistance(descriptor* a,descriptor* b)
{
	float sum=0;
	for(int i=0;i<16;i++)
	{
		sum+=((a->list)[i].sumDx-(b->list)[i].sumDx)*((a->list)[i].sumDx-(b->list)[i].sumDx)
		+((a->list)[i].sumDy-(b->list)[i].sumDy)*((a->list)[i].sumDy-(b->list)[i].sumDy)
		+((a->list)[i].sumAbsDy-(b->list)[i].sumAbsDy)*((a->list)[i].sumAbsDy-(b->list)[i].sumAbsDy)
		+((a->list)[i].sumAbsDx-(b->list)[i].sumAbsDx)*((a->list)[i].sumAbsDx-(b->list)[i].sumAbsDx);
	}
	return sum;
}

