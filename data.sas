/* this first piece merges CRSP/COMPUSTAT */

%INCLUDE "~/git/sas/CC_LINK.sas";
%CC_LINK(dsetin=comp.funda,
    dsetout=compx,
    datevar=datadate,
    keep_vars=at lt);

data crspm6;
    set crsp.msf;
    where month(date)=6;
    ME6=abs(prc*shrout);
    keep permno date ME6;
data crspm;
    set crsp.msf;
    ME=abs(prc*shrout);
    datadate=date;
    keep permno datadate date ME;
run;

/* MERGE_ASOF merges the most recent 
observation in dataset B into dataset A */
%INCLUDE "~/git/sas/MERGE_ASOF.sas";
%MERGE_ASOF(a=crspm,b=crspm6,
    merged=crspm2,
    datevar=date,
    num_vars=ME6);
%MERGE_ASOF(a=crspm2,b=compx,
    merged=crspm3,
    datevar=datadate,
    num_vars=BE ME_COMP at lt gp);
data crspm3;
    set crspm3;
    BM = BE/ME6;
    bm_log = log(BM);
    me_log = log(ME);
run;

proc print data=crspm3(obs=25) width=min;
    where permno=11850 and year(date) between 1993 and 2018;;
var permno date me: bm:;run;

/* This macro creates yearly stock-day files
pulling from both master files and then WRDS IID 
for the second-level TAQ data */
%MACRO TAQ_OWR_GPIN(yyyy=2004);
data work.mastm_&yyyy. ;
    set %if &yyyy > 1993
	%then %do;
	taq.mast_%SYSEVALF(&yyyy.-1):
	%end;
	taq.mast_&yyyy.:
	taq.mast_%SYSEVALF(&yyyy.+1):;
    SYM_ROOT=scan(SYMBOL, 1, ' ');
    SYM_SUFFIX=scan(SYMBOL, 2, ' ');
    DATE=coalesce(FDATE,DATEF);
    format date yymmdd10.;
run;
proc sort data=work.mastm_&yyyy. NODUPKEY;
    by SYMBOL DATE;
run;
proc sql;
    create table work.mastm_crsp_&yyyy. as
	select a.date, sym_root, sym_suffix, symbol,
	substr(coalesce(b.ncusip, b.cusip),1,8) as cusip8,
	a.permno, a.permco, shrcd, exchcd,
	a.prc, a.ret, a.retx, a.shrout, a.vol, c.divamt, c.distcd,
	coalesce(e.SP500,0) as SP500
	from crsp.dsf a
	left join
	crsp.dsenames b
	on a.permno = b.permno
	and a.date between b.namedt and coalesce(b.nameendt, today())
	left join
	crsp.dsedist c
	on a.permno = c.permno
	and a.date = c.paydt
	left join
	(select distinct cusip, sym_root, sym_suffix, symbol,
	min(date) as mindt, max(date) as maxdt
	from work.mastm_&yyyy.
	group by cusip, sym_root, sym_suffix, symbol) d
	on substr(d.cusip,1,8) = substr(coalesce(b.ncusip, b.cusip),1,8)
	and a.date ge d.mindt
	and a.date le coalesce(d.maxdt,today())
	left join
	(select *, 1 as SP500 from crsp.dsp500list) e
	on a.permno = e.permno
	and a.date between e.start and e.ending
	where year(a.date) = &yyyy.
	and symbol is not null
	order by a.date, sym_root, sym_suffix;
quit;
proc sort data=work.mastm_crsp_&yyyy. nodupkey;
    by date sym_root sym_suffix;
run;
proc sort data=taq.wrds_iid_&yyyy.
    out=work.wrds_iid_&yyyy.;
    by date symbol;
run;    
data work.taqdf_&yyyy.;
    length date 8;
    merge work.wrds_iid_&yyyy.(keep=date symbol
	buynumtrades_lri sellnumtrades_lri
	FPrice OPrice CPrc: ret_mkt_t
	vwap_m 
	SumVolume_m SumVolume_b SumVolume_a)
	work.mastm_crsp_&yyyy.;
    by date symbol;
    /* make names consistent with TAQMSEC */
    CCPrc = abs(coalesce(prc,cprc,cprc2));
    mid_after_open = (oprice+fprice)/2;
    y_e = divide(buynumtrades_lri-sellnumtrades_lri,buynumtrades_lri+sellnumtrades_lri);
    symbol_15=symbol;
     rename buynumtrades_lri = n_buys
	sellnumtrades_lri = n_sells
	vwap_m = vw_price_m
	ret_mkt_t = ret_mkt_m
	SumVolume_m = total_vol_m
	SumVolume_b = total_vol_b
	SumVolume_a = total_vol_a;
    label CCPrc='Closing Price (CRSP or TAQ)' y_e='Order Imbalance (%)';
run;
proc sort data=work.taqdf_&yyyy. out=taqdf_&yyyy.x nodupkey;
    by permno date;
    where permno > .Z
	and shrcd in (10,11)
	and exchcd in (1,2,3,4);
run;
%MEND;    

/* This macro creates yearly stock-day files
pulling from both master files and then WRDS IID 
for the millisecond-level TAQ data */
%MACRO TAQM_OWR_GPIN(yyyy=2014);
%let sysyear= %sysfunc(year("&sysdate"d));    
data work.mast1_&yyyy.;
    length date 8 sym_root $6 sym_suffix $10 symbol_15 $15;        
    set taqmsec.mastm_%SYSEVALF(&yyyy.-1):
	taqmsec.mastm_&yyyy.:
	%if %SYSEVALF(&yyyy.+1) <= &sysyear. %then %do;
	taqmsec.mastm_%SYSEVALF(&yyyy.+1):
	%end;;
    SYM_ROOT=scan(SYMBOL_15, 1, ' ');
    SYM_SUFFIX=scan(SYMBOL_15, 2, ' ');
    keep date cusip sym_root sym_suffix symbol_15;
    run;
data work.mast2_&yyyy. ;
    length date 8 sym_root $6 sym_suffix $10 symbol_15 $15;        
    set taq.mast_%SYSEVALF(&yyyy.-1):
	taq.mast_&yyyy.:
	%if %SYSEVALF(&yyyy.+1) <= &sysyear. %then %do;
	taq.mast_%SYSEVALF(&yyyy.+1):
	%end;;        
    SYM_ROOT=scan(SYMBOL, 1, ' ');
    SYM_SUFFIX=scan(SYMBOL, 2, ' ');
    DATE=coalesce(DATE,FDATE,DATEF);
    SYMBOL_15=coalescec(SYMBOL_15,SYMBOL);
    keep date cusip sym_root sym_suffix symbol_15;
run;
data work.mastm_&yyyy.;
    length date 8 cusip $12
	sym_root $6 sym_suffix $10 symbol_15 $15;    
    set work.mast1_&yyyy. work.mast2_&yyyy.;
run;
proc sort data=work.mastm_&yyyy. NODUPKEY;
    by SYM_ROOT SYM_SUFFIX DATE;
run;
proc sql;
    create table work.mastm_crsp_&yyyy. as
	select a.date, sym_root, sym_suffix, symbol_15,
	substr(coalesce(b.ncusip, b.cusip),1,8) as cusip8,
	a.permno, a.permco, shrcd, exchcd,
	a.prc, a.ret, a.retx, a.shrout, a.vol, c.divamt, c.distcd,
	coalesce(e.SP500,0) as SP500
	from crsp.dsf a
	left join
	crsp.dsenames b
	on a.permno = b.permno
	and a.date between b.namedt and coalesce(b.nameendt, today())
	left join
	crsp.dsedist c
	on a.permno = c.permno
	and a.date = c.paydt
	left join
	(select distinct cusip, sym_root, sym_suffix, symbol_15,
	min(date) as mindt, max(date) as maxdt
	from work.mastm_&yyyy.
	group by cusip, sym_root, sym_suffix, symbol_15) d
	on substr(d.cusip,1,8) = substr(coalesce(b.ncusip, b.cusip),1,8)
	and a.date ge d.mindt
	and a.date le coalesce(d.maxdt,today())
	left join
	(select *, 1 as SP500 from crsp.dsp500list) e
	on a.permno = e.permno
	and a.date between e.start and e.ending
	where year(a.date) = &yyyy.
	and symbol_15 is not null
	order by a.date, sym_root, sym_suffix;
quit;
proc sort data=work.mastm_crsp_&yyyy. nodupkey;
    by date sym_root sym_suffix;
run;
proc sort data=taqmsec.wrds_iid_&yyyy.
    out=work.wrds_iid_&yyyy.;
    by date sym_root sym_suffix;
run;        
data work.taqdf_&yyyy.;
    length date 8 sym_root $6 sym_suffix $10;
    merge work.wrds_iid_&yyyy.(keep=date sym_root sym_suffix
	buynumtrades_lr sellnumtrades_lr oprc cprc ret_mkt_m
	vw_price_m mid_after_open
	total_vol_m total_vol_b total_vol_a)
	work.mastm_crsp_&yyyy.;
    by date sym_root sym_suffix;
    CCPrc = abs(coalesce(prc,cprc));
    y_e = divide(buynumtrades_lr-sellnumtrades_lr,buynumtrades_lr+sellnumtrades_lr);
    rename buynumtrades_lr=n_buys sellnumtrades_lr=n_sells;
    label CCPrc='Closing Price (CRSP or TAQ)' y_e='Order Imbalance (%)';
run;
proc sort data=work.taqdf_&yyyy. out=taqdf_&yyyy.x nodupkey;
    by permno date;
    where permno > .Z
	and shrcd in (10,11)
	and exchcd in (1,2,3,4);
run;
%MEND;

%TAQ_OWR_GPIN(yyyy=1993);
%TAQ_OWR_GPIN(yyyy=1994);
%TAQ_OWR_GPIN(yyyy=1995);
%TAQ_OWR_GPIN(yyyy=1996);
%TAQ_OWR_GPIN(yyyy=1997);
%TAQ_OWR_GPIN(yyyy=1998);
%TAQ_OWR_GPIN(yyyy=1999);
%TAQ_OWR_GPIN(yyyy=2000);
%TAQ_OWR_GPIN(yyyy=2001);
%TAQ_OWR_GPIN(yyyy=2002);
%TAQ_OWR_GPIN(yyyy=2003);
%TAQ_OWR_GPIN(yyyy=2004);
%TAQ_OWR_GPIN(yyyy=2005);
%TAQ_OWR_GPIN(yyyy=2006);
/* NMS Implementation Feb 2007 */
%TAQM_OWR_GPIN(yyyy=2007);
%TAQM_OWR_GPIN(yyyy=2008);
%TAQM_OWR_GPIN(yyyy=2009);
%TAQM_OWR_GPIN(yyyy=2010);
%TAQM_OWR_GPIN(yyyy=2011);
%TAQM_OWR_GPIN(yyyy=2012);
%TAQM_OWR_GPIN(yyyy=2013);
%TAQM_OWR_GPIN(yyyy=2014);
%TAQM_OWR_GPIN(yyyy=2015);
%TAQM_OWR_GPIN(yyyy=2016);
%TAQM_OWR_GPIN(yyyy=2017);
%TAQM_OWR_GPIN(yyyy=2018);
%TAQM_OWR_GPIN(yyyy=2019);

data taqdfx_all;
    set taqdf_:;
run;

proc sql;
    create table taqdfx_all1 as
	select a.*, b.vwretd, b.vwretx
	from taqdfx_all a
	left join crsp.dsiy b
	on a.date = b.caldt
	order by a.permno, a.date;
quit;

/* Compute and adjust OWR variables */
proc printto log='/dev/null';run;
proc expand data=taqdfx_all1
    out=taqdfx_all2
    method=none;
    by permno;
    convert y_e = y_eL1 / transformout = (lag 1);
    convert ccprc = CCPrcL1 / transformout = (lag 1);
    convert mid_after_open = omF1 / transformout = (lead 1);
run;
proc printto;run;
%put expand &syslast. done;

data taqdfx_all2;
    set taqdfx_all2;
    yyyy=year(date);
    r_d = (vw_price_m-mid_after_open+coalesce(divamt,0))/mid_after_open;
    r_o = (omF1-vw_price_m)/mid_after_open;
run;

%MERGE_ASOF(a=taqdfx_all2,b=crspm3,
    merged=taqdfx_all3,
    datevar=date,
    num_vars=bm_log me_log);

proc printto log='/dev/null';run;
proc reg data=taqdfx_all3 outest=_beta
    (drop=_: retx rename=(Intercept=alpha vwretx=beta)) noprint;
    by permno yyyy;
    model retx = vwretx;
run;
proc printto;run;

data taqdfx_all4;
    merge taqdfx_all3 _beta;
    by permno yyyy;
run;
proc sort data=taqdfx_all4 nodupkey;
    by date permno;
run;

proc printto log='/dev/null';run;
proc reg data=taqdfx_all4 noprint;
      model r_o r_d = beta me_log bm_log;
      output out=_ret_resid(keep=permno date ur_o ur_d) r=ur_o ur_d;
      model y_e = y_eL1 me_log;
      output out=_oib_resid(keep=permno date uy_e) r=uy_e;
      by date;
run;
proc printto;run;

data taqdfx_all5;
    merge taqdfx_all4 _ret_resid _oib_resid;
    by date permno;
run;

%INCLUDE "~/git/sas/WINSORIZE_TRUNCATE.sas";
%WINSORIZE_TRUNCATE(dsetin = taqdfx_all5, 
    dsetout = taqdfx_all6, 
    byvar = date, 
    vars = ur_o ur_d, 
    type = W, 
    pctl = 1 99,
    filter = and exchcd eq 1);

/* Output files */
proc sort data=taqdfx_all6
    out=out.taqdfx_all6(compress=no) nodupkey;
    by permno date;
proc sort data=crspm3
    out=out.crspm3 nodupkey;
    by permno date;
run;