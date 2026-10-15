#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
inline double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
inline void require(bool b,const std::string&s){if(!b)throw std::runtime_error(s);}
struct Data{size_t n,d;std::vector<float> x;};
inline Data fvecs(const fs::path&p){std::ifstream f(p,std::ios::binary);int d;f.read((char*)&d,4);require(bool(f)&&d>0,"fvecs header");Data a{fs::file_size(p)/((d+1)*4),size_t(d),{}};a.x.resize(a.n*a.d);f.seekg(0);for(size_t i=0;i<a.n;i++){int dd;f.read((char*)&dd,4);require(dd==d,"fvecs dims");f.read((char*)(a.x.data()+i*d),d*4);}require(bool(f),"fvecs truncated");return a;}
inline Data metadata(const fs::path&p){std::ifstream f(p,std::ios::binary);Data a;f.read((char*)&a.n,8);f.read((char*)&a.d,8);require(bool(f)&&a.n>0&&a.d>=2&&a.d<=4,"metadata header");a.x.resize(a.n*a.d);f.read((char*)a.x.data(),a.x.size()*4);require(bool(f),"metadata truncated");return a;}
// Binary format explicit fields, no struct padding. Float32 predicate semantics.
struct Predicate{std::array<float,4> c{},lo{},hi{};float r=0;};
inline float distance2(const float*m,const Predicate&p,int d){float v=0;for(int j=0;j<d;j++){float x=m[j]-p.c[j];v+=x*x;}return v;}
inline bool inbox(const float*m,const Predicate&p,int d){for(int j=0;j<d;j++)if(m[j]<p.lo[j]||m[j]>p.hi[j])return false;return true;}
inline bool qualifies(const float*m,const Predicate&p,int d,int kind){bool ball=distance2(m,p,d)<=p.r*p.r;return kind==0?ball:inbox(m,p,d)&&!ball;}
struct Workload{uint64_t nq=0,d=0,kind=0,k=20;double ratio=0;std::vector<Predicate> p;std::vector<uint64_t> counts;std::vector<int64_t> gt;};
inline void save(const fs::path&path,const Workload&w){require(!fs::exists(path),"overwrite workload");std::ofstream f(path,std::ios::binary);f.write("RCWL0001",8);for(auto v:{w.nq,w.d,w.kind,w.k})f.write((char*)&v,8);f.write((char*)&w.ratio,8);for(auto&p:w.p){f.write((char*)p.c.data(),16);f.write((char*)p.lo.data(),16);f.write((char*)p.hi.data(),16);f.write((char*)&p.r,4);}f.write((char*)w.counts.data(),w.nq*8);f.write((char*)w.gt.data(),w.gt.size()*8);require(bool(f),"save workload");}
inline Workload load(const fs::path&path){std::ifstream f(path,std::ios::binary);char magic[8];f.read(magic,8);require(std::string(magic,8)=="RCWL0001","workload magic");Workload w;for(auto v:{&w.nq,&w.d,&w.kind,&w.k})f.read((char*)v,8);f.read((char*)&w.ratio,8);require(w.nq>0&&w.nq<=1000&&w.d>=2&&w.d<=4&&w.kind<=1&&w.k==20,"workload header");w.p.resize(w.nq);w.counts.resize(w.nq);w.gt.resize(w.nq*w.k);for(auto&p:w.p){f.read((char*)p.c.data(),16);f.read((char*)p.lo.data(),16);f.read((char*)p.hi.data(),16);f.read((char*)&p.r,4);}f.read((char*)w.counts.data(),w.nq*8);f.read((char*)w.gt.data(),w.gt.size()*8);require(bool(f)&&f.peek()==EOF,"workload size");return w;}
inline float order(std::vector<float>&v,size_t n){require(n>0&&n<=v.size(),"order statistic");std::nth_element(v.begin(),v.begin()+n-1,v.end());return v[n-1];}
inline float radius(float d2){float r=std::sqrt(d2);while(r*r<d2)r=std::nextafter(r,INFINITY);return r;}
inline double percentile(std::vector<double>v,double q){std::sort(v.begin(),v.end());return v[size_t(q*(v.size()-1)+.5)];}
