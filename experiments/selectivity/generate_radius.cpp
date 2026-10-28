#include "common.h"
int main(int argc,char**argv){try{
 require(argc==6,"generate DATA DIM KIND NQ OUTDIR");fs::path root=argv[1],out=argv[5];int d=std::stoi(argv[2]),kind=std::stoi(argv[3]);size_t nq=std::stoul(argv[4]);require(kind==0||d==2,"compose only 2d");fs::create_directories(out);
 auto X=fvecs(root/"sift_base.fvecs"),Q=fvecs(root/"sift_query.fvecs"),M=metadata(root/("sift_metadata_uniform_"+std::to_string(d)+"d.bin"));require(M.n==X.n&&M.d==size_t(d)&&nq<=Q.n,"shape mismatch");
 for(float v:X.x)require(v==std::floor(v)&&v>=0&&v<=255,"SIFT integer exact-distance assumption");
 for(float v:Q.x)require(v==std::floor(v)&&v>=0&&v<=255,"query integer assumption");
 for(float v:M.x)require(v>=0&&v<=1,"raw metadata outside unit domain");
 for(double pct:{0.5}){auto t=Clock::now();Workload w;w.nq=nq;w.d=d;w.kind=kind;w.ratio=pct/100.;w.p.resize(nq);w.counts.resize(nq);w.gt.assign(nq*20,-1);std::vector<double>prep(nq);std::vector<uint64_t>outer(nq);size_t target=std::llround(X.n*w.ratio);
 #pragma omp parallel for schedule(dynamic,1)
 for(size_t qi=0;qi<nq;qi++){auto t0=Clock::now();auto&p=w.p[qi];std::mt19937 rng(20261010u+qi*97u+d*100003u+kind*1000003u);std::uniform_real_distribution<float> u(kind==0?0.f:.2f,kind==0?1.f:.8f);for(int j=0;j<d;j++)p.c[j]=u(rng);
 std::vector<float> ds(M.n);if(kind==0){for(size_t i=0;i<M.n;i++)ds[i]=distance2(M.x.data()+i*d,p,d);p.r=radius(order(ds,target));}
 else{for(size_t i=0;i<M.n;i++){float v=0;for(int j=0;j<d;j++)v=std::max(v,std::abs(M.x[i*d+j]-p.c[j]));ds[i]=v;}float h=order(ds,size_t(std::ceil(target/.7)));h=std::nextafter(h,INFINITY);for(int j=0;j<d;j++){p.lo[j]=p.c[j]-h;p.hi[j]=p.c[j]+h;}ds.clear();for(size_t i=0;i<M.n;i++)if(inbox(M.x.data()+i*d,p,d))ds.push_back(distance2(M.x.data()+i*d,p,d));outer[qi]=ds.size();p.r=radius(order(ds,ds.size()-target));}
 prep[qi]=elapsed(t0);std::priority_queue<std::pair<double,int64_t>> heap;uint64_t count=0;
 for(size_t i=0;i<M.n;i++){if(!qualifies(M.x.data()+i*d,p,d,kind))continue;count++;double dist=0;for(size_t j=0;j<X.d;j++){double diff=double(Q.x[qi*Q.d+j])-X.x[i*X.d+j];dist+=diff*diff;}std::pair<double,int64_t> c{dist,int64_t(i)};if(heap.size()<20)heap.push(c);else if(c<heap.top()){heap.pop();heap.push(c);}}
 w.counts[qi]=count;size_t n=heap.size();while(!heap.empty()){w.gt[qi*20+--n]=heap.top().second;heap.pop();}
 }
 std::string stem=(kind?"compose":"radius")+std::to_string(d)+"d_"+std::string("0050");require(!fs::exists(out/(stem+".bin")),"refusing to overwrite workload");save(out/(stem+".bin"),w);std::ofstream csv(out/(stem+".queries.csv"));csv<<std::setprecision(10)<<"query_id,eligible,outer_count,radius,c0,c1,c2,c3,lo0,lo1,hi0,hi1,calibration_seconds\n";for(size_t qi=0;qi<nq;qi++){auto&p=w.p[qi];csv<<qi<<','<<w.counts[qi]<<','<<outer[qi]<<','<<p.r;for(auto c:p.c)csv<<','<<c;csv<<','<<p.lo[0]<<','<<p.lo[1]<<','<<p.hi[0]<<','<<p.hi[1]<<','<<prep[qi]<<'\n';}std::cout<<stem<<" seconds="<<elapsed(t)<<" mincount="<<*std::min_element(w.counts.begin(),w.counts.end())<<std::endl;
 }
}catch(std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
