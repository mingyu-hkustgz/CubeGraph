// Same frozen predicates/GT, existing production indexes, single-thread search.
// Compile separately for CG/POST and ACORN to avoid library symbol collisions.
#include "common.h"
#ifdef ACORN_DRIVER
#include <faiss/IndexACORN.h>
#include <faiss/index_io.h>
#else
#define USE_SSE
#define USE_AVX
#define USE_AVX512
#include "IndexCube.h"
#include "config.h"
#include "hnsw-static.h"
#endif
int main(int argc,char**argv){try{
 require(argc==7,"benchmark METHOD DATA INDEX WORKLOAD OUTCSV SMOKE(0/1)");std::string method=argv[1];fs::path data=argv[2],idx=argv[3],wlpath=argv[4],out=argv[5];bool smoke=std::stoi(argv[6]);require(!fs::exists(out),"output exists");auto w=load(wlpath);auto X=fvecs(data/"sift_base.fvecs"),Q=fvecs(data/"sift_query.fvecs"),M=metadata(data/("sift_metadata_uniform_"+std::to_string(w.d)+"d.bin"));require(X.n==M.n&&M.d==w.d&&Q.n>=w.nq,"data mismatch");
 if(!smoke)for(auto count:w.counts)require(count>=20,"formal workload has fewer than k eligible points");
 omp_set_num_threads(1);
 auto loading=Clock::now();uint64_t index_bytes=fs::file_size(idx);
#ifdef ACORN_DRIVER
 require(method=="acorn","method mismatch");auto*index=dynamic_cast<faiss::IndexACORNFlat*>(faiss::read_index(idx.c_str()));require(index&&index->d==X.d&&index->ntotal==X.n,"ACORN index mismatch");
#else
 require(method=="cg"||method=="post","method mismatch");IndexCube*cg=nullptr;hnswlib::HierarchicalNSWStatic<float>*post=nullptr;hnswlib::L2Space space(X.d);if(method=="cg"){hnswlib::HierarchicalNSWCube<float>::static_base_data_=(char*)X.x.data();cg=new IndexCube(6,16,200,2);std::string xp=(data/"sift_base.fvecs").string();require(cg->load_index(idx.string(),xp.data()),"load cube");require(cg->get_meta_dim()==w.d,"cube dimensions");auto&im=cg->get_metadata();require(im.size()==M.n,"cube n");for(size_t i=0;i<M.n;i++)for(size_t j=0;j<M.d;j++)require(im[i][j]==M.x[i*M.d+j],"cube metadata mismatch");for(int l=0;;l++){auto p=idx.string()+".layer_"+std::to_string(l);if(!fs::exists(p))break;index_bytes+=fs::file_size(p);}}
 else{hnswlib::HierarchicalNSWStatic<float>::static_base_data_=(char*)X.x.data();post=new hnswlib::HierarchicalNSWStatic<float>(&space,idx.string());}
#endif
 double load_seconds=elapsed(loading);std::vector<double>prep_us(w.nq);std::vector<char>masks;
#ifndef ACORN_DRIVER
 std::vector<RadiusFilterParams> balls;std::vector<CompositeFilterParams>comps;balls.reserve(w.nq);comps.reserve(w.nq);
#endif
 for(size_t qi=0;qi<w.nq;qi++){
 auto t=Clock::now();const auto&p=w.p[qi];
#ifdef ACORN_DRIVER
 if(qi==0)masks.resize(w.nq*X.n);char*mask=masks.data()+qi*X.n;uint64_t cnt=0;for(size_t i=0;i<X.n;i++){mask[i]=qualifies(M.x.data()+i*w.d,p,w.d,w.kind);cnt+=mask[i];}require(cnt==w.counts[qi],"mask count mismatch");
#else
 if(method=="cg"){std::vector<float>c(p.c.begin(),p.c.begin()+w.d);balls.emplace_back(c,p.r,w.d);CompositeFilterParams cp;cp.attr_dim=w.d;cp.radius=balls.back();cp.bbox=BoundingBox(w.d);for(size_t j=0;j<w.d;j++){cp.bbox.min_bounds[j]=p.lo[j];cp.bbox.max_bounds[j]=p.hi[j];}cp.spec.use_bbox=true;cp.spec.use_radius=true;cp.spec.radius_negate=true;comps.push_back(cp);}
#endif
 prep_us[qi]=elapsed(t)*1e6;
 }
 double prepmean=std::accumulate(prep_us.begin(),prep_us.end(),0.)/w.nq;
 std::ofstream csv(out);csv<<std::setprecision(12)<<"dataset,dimension,predicate,target_ratio,measured_selectivity,queries,k,method,search_width,repeat,recall,qps,p50_us,p95_us,prep_us_mean,prep_us_p95,index_bytes,load_seconds,search_seconds\n";
 auto search=[&](size_t qi,unsigned ef,int64_t*ids){
#ifdef ACORN_DRIVER
 float distances[20];index->search(1,Q.x.data()+qi*Q.d,20,distances,ids,masks.data()+qi*X.n);
#else
 std::priority_queue<std::pair<float,hnswlib::labeltype>> result;
 if(cg)result=w.kind==0?cg->fly_search(Q.x.data()+qi*Q.d,20,&balls[qi]):cg->fly_search(Q.x.data()+qi*Q.d,20,&comps[qi]);
 else{auto cand=post->searchKnn(Q.x.data()+qi*Q.d,ef);while(!cand.empty()){auto v=cand.top();cand.pop();require(v.second<X.n,"invalid POST candidate");if(qualifies(M.x.data()+v.second*M.d,w.p[qi],w.d,w.kind)){result.push(v);if(result.size()>20)result.pop();}}}
 size_t i=0;while(!result.empty()){ids[i++]=result.top().second;result.pop();}
#endif
 };
 for(int rep=1;rep<=(smoke?1:3);rep++)for(unsigned ef:std::vector<unsigned>{20,32,64,128,256,512,1024,2048}){if(smoke&&ef!=20&&ef!=2048)continue;
#ifdef ACORN_DRIVER
 index->acorn.efSearch=ef;
#else
 if(cg)cg->set_global_ef(ef);
#endif
 for(size_t qi=0;qi<std::min<uint64_t>(20,w.nq);qi++){int64_t ids[20];std::fill(ids,ids+20,-1);search(qi,ef,ids);}
 std::vector<int64_t>ids(w.nq*20,-1);std::vector<double>lat(w.nq);auto t=Clock::now();for(size_t qi=0;qi<w.nq;qi++){auto q0=Clock::now();search(qi,ef,ids.data()+qi*20);lat[qi]=elapsed(q0)*1e6;}double seconds=elapsed(t);
 // Audit and artifact IO are outside the timed region for all three methods.
 std::string idpath=out.string()+".ef"+std::to_string(ef)+".rep"+std::to_string(rep)+".ids.bin";std::ofstream idfile(idpath,std::ios::binary);idfile.write((char*)ids.data(),ids.size()*8);require(bool(idfile),"ID write failed");idfile.close();
 uint64_t hits=0,denom=0;for(size_t qi=0;qi<w.nq;qi++){std::vector<int64_t>seen;for(int k=0;k<20;k++){auto id=ids[qi*20+k];if(id<0)continue;require(size_t(id)<M.n&&qualifies(M.x.data()+id*M.d,w.p[qi],w.d,w.kind),"invalid returned ID or predicate violation: query="+std::to_string(qi)+" id="+std::to_string(id)+" eligible="+std::to_string(w.counts[qi]));require(std::find(seen.begin(),seen.end(),id)==seen.end(),"duplicate ID");seen.push_back(id);hits+=std::find(w.gt.begin()+qi*20,w.gt.begin()+(qi+1)*20,id)!=w.gt.begin()+(qi+1)*20;}for(int k=0;k<20;k++)denom+=w.gt[qi*20+k]>=0;}

 double sel=double(std::accumulate(w.counts.begin(),w.counts.end(),uint64_t(0)))/(w.nq*double(M.n));double recall=denom?double(hits)/denom:1.;csv<<"sift,"<<w.d<<','<<(w.kind?"compose":"radius")<<','<<w.ratio<<','<<sel<<','<<w.nq<<",20,"<<method<<','<<ef<<','<<rep<<','<<recall<<','<<w.nq/seconds<<','<<percentile(lat,.5)<<','<<percentile(lat,.95)<<','<<prepmean<<','<<percentile(prep_us,.95)<<','<<index_bytes<<','<<load_seconds<<','<<seconds<<'\n';csv.flush();std::cout<<method<<" d="<<w.d<<" kind="<<w.kind<<" ratio="<<w.ratio<<" rep="<<rep<<" ef="<<ef<<" recall="<<recall<<" qps="<<w.nq/seconds<<std::endl;
 }
#ifdef ACORN_DRIVER
 delete index;
#else
 delete cg;delete post;
#endif
}catch(std::exception&e){std::cerr<<"ERROR "<<e.what()<<'\n';return 1;}}
