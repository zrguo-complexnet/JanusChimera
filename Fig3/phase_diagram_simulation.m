
%% phase_diagram_simulation.m
% Numerical phase diagrams for the n=5 Kuramoto model on S^2 (Delta=0).
%
% This script INTEGRATES the N-oscillator equations and measures time-
% averaged azimuthal coherence |Z_C1| and |Z_C2| relative to the
% instantaneous mean-field direction rho/|rho|.
%
% Reproducibility:
%   1. Set cfg.mode='simulate' to run the parameter scan (default).
%   2. Set cfg.mode='plot' to redraw existing numerical phase maps.
%   3. Run three_cluster_angle_boundaries_rho.m separately to compute
%      the corrected theoretical boundaries and overlay them on the maps.
%
% The simulation deliberately DOES NOT calculate theoretical boundaries.
% No physical model, initial-condition, integration or convergence formula
% has been changed from the rho-axis RK45 CPU/GPU simulation (engine v5).
% See README.md for the data files, modes and optional GPU support.

clear; clc;

%% Configuration

% Modes: 'simulate' = recompute data; 'plot' = reload existing data;
%        'auto' = reload compatible data if found, otherwise simulate.

cfg.mode = 'simulate';
cfg.checkVersion = 3;
cfg.engineVersion = 5;

cfg.integrator = 'rk45';
cfg.backend = 'auto';  % 'auto', 'cpu', or 'gpu'

cfg.gpuBatchSize = 1024;
cfg.gpuMemoryFraction = 0.35;

cfg.relTol = 1e-6;
cfg.absTol = 1e-8;

cfg.N = 1000;
cfg.K = 50;
cfg.n = 5;
cfg.Delta = 0;

cfg.gamma1 = 0:0.01:0.50;
cfg.gamma2 = 0:0.01:0.35;

cfg.perturbation = 1e-3;
cfg.seedBase = 20260919;
cfg.seedOffset = 0;

cfg.dtMax = 2.0;
cfg.phaseStep = 0.30;
cfg.phaseGuard = 0.30;
cfg.feedbackStep = 0.50;
cfg.stepFactor = 1;

cfg.sampleDt = 0.25;
cfg.burnIn = 6000;
cfg.window = 2000;
cfg.tMax = 30000;

cfg.meanTol = 0.005;
cfg.requiredPasses = 2;
cfg.minWindowClock = 20;
cfg.allowEarlyStop = true;

cfg.fluctuationTol = 0.002;
cfg.pCoherenceThreshold = 0.99;

cfg.useParallel = true;
cfg.resume = false;

cfg.progressSeconds = 15;
cfg.checkpointSeconds = 120;

% Plotting and exports (the separate theory script adds the boundaries).
cfg.fontName = 'Times New Roman';
cfg.tickFont = 24;
cfg.labelFont = 28;
cfg.saveFigures = true;
cfg.drawDiagnostics = true;
cfg.writeCsv = true;

rootDir = fileparts(mfilename('fullpath'));
if isempty(rootDir)
    rootDir = pwd;
end

cfg.outputDir = fullfile(rootDir, ...
    'matlab_output_gpu_batch_rho_axis');

dataFile = fullfile(rootDir, ...
    'chimera_phase_gpu_batch_rho_axis_results.mat');

assert(cfg.n==5 && cfg.Delta==0 && cfg.K>0, ...
    'This script implements n=5, Delta=0, K>0.');

validateConfiguration(cfg);

if ~exist(cfg.outputDir,'dir')
    mkdir(cfg.outputDir);
end

fprintf('Pure MATLAB | %s | %s | N=%d | %d points.\n', ...
    cfg.integrator,cfg.backend,cfg.N, ...
    numel(cfg.gamma1)*numel(cfg.gamma2));

fprintf('Azimuthal reference axis: instantaneous rho.\n');

%% Load or simulate

loadExisting = strcmpi(cfg.mode,'plot') || ...
    (strcmpi(cfg.mode,'auto') && exist(dataFile,'file')==2);

if ~loadExisting
    cfg = selectBackend(cfg);
end

if loadExisting
    assert(exist(dataFile,'file')==2, ...
        'Numerical results not found. Run with cfg.mode=''simulate'' first.');

    % Older MAT files may contain a legacy 'theory' field; it is ignored.
    loaded = load(dataFile,'results','simulationCfg');
    assert(isfield(loaded,'simulationCfg') && ...
        isfield(loaded,'results') && ...
        isfield(loaded.results,'meanCheckPassed') && ...
        isfield(loaded.simulationCfg,'checkVersion') && ...
        isfield(loaded.simulationCfg,'engineVersion') && ...
        loaded.simulationCfg.checkVersion==cfg.checkVersion && ...
        loaded.simulationCfg.engineVersion==cfg.engineVersion && ...
        'Old-format numerical data cannot be reused.');

    % Even in plot-only mode, never silently use the wrong physical model.
    stored = loaded.simulationCfg;
    assert(stored.n==cfg.n && ...
        stored.K==cfg.K && stored.Delta==cfg.Delta, ...
        'Saved data use different model parameters.');

    if ~strcmpi(cfg.mode,'plot')
        fields = simulationFields();
        for k=1:numel(fields)
            key=fields{k};
            assert(isfield(stored,key) && ...
                compatibleValue(stored.(key),cfg.(key)), ...
                'Saved configuration differs (%s).',key);
        end
    end

    results = loaded.results;

else
    simulationCfg = cfg;
    pointResults = scanGrid(cfg);
    results = packResults(pointResults,cfg);

    % Theory is intentionally absent: boundary calculations are independent.
    save(dataFile,'results','simulationCfg','-v7');
end

%% Numerical maps and diagnostics (without theoretical overlays)
for group=1:2
    drawMap(results,cfg,group);
end

if cfg.drawDiagnostics
    drawDiagnostics(results,cfg);
end
if cfg.writeCsv
    writeSummary(results,cfg);
end

fprintf('Mean-converged: %d; unresolved: %d; no ring: %d.\n', ...
    nnz(results.status==1), ...
    nnz(results.status==0), ...
    nnz(results.status==-1));

fprintf('Time averages of |Z| over the final 2*window = %.0f time units.\n', ...
    2*cfg.window);
fprintf('Azimuthal reference axis: instantaneous rho.\n');
fprintf('Results MAT: %s\n',dataFile);
fprintf('Figure/CSV directory: %s\n',cfg.outputDir);


%% ============================================================
%  Configuration validation
% =============================================================

function validateConfiguration(cfg)

assert(any(strcmpi(cfg.mode, ...
    {'auto','simulate','plot'})), 'Unknown mode.');

assert(strcmpi(cfg.integrator,'rk45'), ...
    'This script uses RK45.');

assert(any(strcmpi(cfg.backend, ...
    {'auto','cpu','gpu'})), 'Unknown backend.');

assert(cfg.gpuBatchSize>=1 && ...
    cfg.gpuBatchSize==round(cfg.gpuBatchSize) && ...
    cfg.gpuMemoryFraction>0 && ...
    cfg.gpuMemoryFraction<=0.5, ...
    'Invalid GPU settings.');

assert(cfg.relTol>0 && cfg.absTol>0 && ...
    cfg.phaseGuard>0, 'Invalid RK45 settings.');

assert(cfg.phaseStep<=cfg.phaseGuard && ...
    cfg.phaseGuard<=0.30, ...
    'Invalid phase guard.');

assert(cfg.feedbackStep>0 && ...
    cfg.feedbackStep<=0.50, ...
    'Invalid feedback guard.');

assert(cfg.progressSeconds>0 && ...
    cfg.checkpointSeconds>0);

assert(cfg.N>=1 && cfg.N==round(cfg.N));

assert(~isempty(cfg.gamma1) && ...
    ~isempty(cfg.gamma2) && ...
    all(isfinite(cfg.gamma1)) && ...
    all(isfinite(cfg.gamma2)) && ...
    all(cfg.gamma1>=0) && ...
    all(cfg.gamma2>=0));

assert(round(cfg.N*max(cfg.gamma1))+ ...
    round(cfg.N*max(cfg.gamma2))<cfg.N, ...
    'A nonempty P population is required.');

assert(cfg.sampleDt>0 && ...
    cfg.window>=cfg.sampleDt && ...
    cfg.burnIn>=0 && ...
    cfg.dtMax>0 && ...
    cfg.phaseStep>0 && ...
    cfg.stepFactor>0);

assert(cfg.tMax>=cfg.burnIn+2*cfg.window);

ratios=[cfg.burnIn,cfg.window,cfg.tMax]/cfg.sampleDt;

assert(all(abs(ratios-round(ratios))<1e-8), ...
    'Time settings must be multiples of sampleDt.');

assert(abs((cfg.tMax-cfg.burnIn)/cfg.window- ...
    round((cfg.tMax-cfg.burnIn)/cfg.window))<1e-8);

assert(cfg.requiredPasses>=1 && ...
    cfg.requiredPasses==round(cfg.requiredPasses));

assert(cfg.meanTol>0 && ...
    cfg.minWindowClock>=0 && ...
    cfg.fluctuationTol>=0);

assert(cfg.pCoherenceThreshold>0 && ...
    cfg.pCoherenceThreshold<=1);

end


%% ============================================================
%  Backend
% =============================================================

function cfg=selectBackend(cfg)

cfg.executionBackend='cpu';
cfg.gpuBatchSizeEffective=0;

if strcmpi(cfg.backend,'cpu')
    return;
end

problemText='GPU unavailable.';

if license('test','Distrib_Computing_Toolbox') && ...
        exist('gpuDevice','file')~=0

    try

        dev=gpuDevice;

        bytesPerPoint=8*cfg.N*3*48;

        memorySlots=floor( ...
            cfg.gpuMemoryFraction* ...
            double(dev.AvailableMemory)/bytesPerPoint);

        assert(memorySlots>=1, ...
            'Insufficient GPU memory.');

        cfg.gpuBatchSizeEffective= ...
            min(cfg.gpuBatchSize,memorySlots);

        cfg.executionBackend='gpu';

        fprintf('GPU: %s; batch size up to %d.\n', ...
            dev.Name,cfg.gpuBatchSizeEffective);

        return;

    catch problem
        problemText=problem.message;
    end
end

if strcmpi(cfg.backend,'gpu')
    error('Requested GPU unavailable: %s',problemText);
end

fprintf('GPU unavailable; using CPU. %s\n',problemText);

end


function yes=compatibleValue(a,b)

if isnumeric(a) && isnumeric(b)

    yes=isequal(size(a),size(b)) && ...
        all(abs(double(a(:))-double(b(:))) <= ...
        1e-12*max(1,max(abs(double(b(:))))));

else
    yes=isequaln(a,b);
end

end


function fields=simulationFields()

fields={'checkVersion','engineVersion', ...
    'integrator','relTol','absTol', ...
    'N','K','n','Delta','gamma1','gamma2', ...
    'perturbation','seedBase','seedOffset', ...
    'dtMax','phaseStep','phaseGuard', ...
    'feedbackStep','stepFactor', ...
    'sampleDt','burnIn','tMax','window', ...
    'requiredPasses','minWindowClock','meanTol', ...
    'allowEarlyStop','fluctuationTol', ...
    'pCoherenceThreshold'};

end


function yes=sameSimulation(a,b)

fields=simulationFields();
yes=true;

for k=1:numel(fields)

    key=fields{k};

    if ~isfield(a,key) || ...
            ~compatibleValue(a.(key),b.(key))
        yes=false;
        return;
    end

end

end


%% ============================================================
%  Parameter scanning
% =============================================================

function pointResults=scanGrid(cfg)

nx=numel(cfg.gamma1);
ny=numel(cfg.gamma2);
total=nx*ny;

simulationCfg=cfg;
pointResults=cell(ny,nx);

checkpointFile=fullfile(cfg.outputDir, ...
    'scan_checkpoint_gpu_batch.mat');

if cfg.resume && exist(checkpointFile,'file')

    old=load(checkpointFile, ...
        'pointResults','simulationCfg');

    if isfield(old,'simulationCfg') && ...
            sameSimulation(old.simulationCfg,cfg) && ...
            isfield(old,'pointResults') && ...
            isequal(size(old.pointResults),[ny,nx])

        pointResults=old.pointResults;

        fprintf('Resuming %d/%d points.\n', ...
            nnz(~cellfun(@isempty,pointResults)),total);

    else
        fprintf('Incompatible checkpoint; starting anew.\n');
    end
end

pending=find(cellfun(@isempty,pointResults));

if isempty(pending)
    return;
end

done=total-numel(pending);

scanClock=tic;
progressClock=tic;
saveClock=tic;

runningTime=zeros(total,1);
runningSteps=zeros(total,1);
latestPoint=0;

futures=[];
futurePoints=[];
serialPoint=0;
gpuPoints=[];

cleanupState=containers.Map( ...
    'KeyType','char','ValueType','any');

cleanupState('checkpointFile')=checkpointFile;
cleanupState('simulationCfg')=simulationCfg;
cleanupState('pointResults')=pointResults;
cleanupState('futures')=[];
cleanupState('finished')=false;

cleanup=onCleanup(@()cleanupScan(cleanupState)); %#ok<NASGU>

parallelOK=cfg.useParallel && ...
    strcmpi(cfg.executionBackend,'cpu') && ...
    license('test','Distrib_Computing_Toolbox') && ...
    exist('parpool','file')~=0;

if strcmpi(cfg.executionBackend,'gpu')

    fprintf('GPU batch scanning started.\n');

    reportProgress(true);

    simulateGpuBatch(pending,cfg,@receiveGpuEvent);

elseif parallelOK

    pool=gcp('nocreate');

    if isempty(pool)
        fprintf('Starting CPU parallel pool...\n');
        pool=parpool;
    end

    queue=parallel.pool.DataQueue;

    afterEach(queue,@receiveProgress);

    futures=parallel.FevalFuture.empty(0,1);

    next=1;
    slots=min(pool.NumWorkers,numel(pending));

    fprintf('CPU workers: %d.\n',slots);

    for k=1:slots
        submitNext();
    end

    reportProgress(true);

    while ~isempty(futures)

        [idx,value]=fetchNext( ...
            futures,cfg.progressSeconds);

        if ~isempty(idx)

            pointResults{futurePoints(idx)}=value;
            done=done+1;

            cleanupState('pointResults')=pointResults;

            futures(idx)=[];
            futurePoints(idx)=[];

            cleanupState('futures')=futures;

            if next<=numel(pending)
                submitNext();
            end
        end

        reportProgress(false);

        if toc(saveClock)>=cfg.checkpointSeconds
            saveCheckpoint();
        end

    end

else

    fprintf('Serial CPU execution.\n');

    for k=1:numel(pending)

        serialPoint=pending(k);

        [j,i]=ind2sub([ny,nx],serialPoint);

        pointResults{serialPoint}=simulatePoint( ...
            cfg.gamma1(i),cfg.gamma2(j),cfg, ...
            @receiveProgress,serialPoint);

        cleanupState('pointResults')=pointResults;

        done=done+1;
        serialPoint=0;

        reportProgress(false);

        if toc(saveClock)>=cfg.checkpointSeconds
            saveCheckpoint();
        end

    end
end

reportProgress(true);
saveCheckpoint();

cleanupState('finished')=true;


    function submitNext()

        id=pending(next);
        next=next+1;

        [row,col]=ind2sub([ny,nx],id);

        futures(end+1,1)=parfeval( ...
            pool,@simulatePoint,1, ...
            cfg.gamma1(col),cfg.gamma2(row), ...
            cfg,queue,id);

        futurePoints(end+1,1)=id;

        cleanupState('futures')=futures;

    end


    function receiveProgress(msg)

        id=msg.pointID;

        if id>=1 && id<=total

            runningTime(id)=msg.t;
            runningSteps(id)=msg.steps;
            latestPoint=id;

        end

        if ~parallelOK

            reportProgress(false);

            if toc(saveClock)>=cfg.checkpointSeconds
                saveCheckpoint();
            end

        end

    end


    function receiveGpuEvent(msg)

        switch msg.kind

            case 'active'

                gpuPoints=msg.pointIDs;

            case 'progress'

                ids=msg.pointIDs;

                runningTime(ids)=msg.t;
                runningSteps(ids)=msg.steps;

                if ~isempty(ids)
                    latestPoint=ids(end);
                end

            case 'complete'

                id=msg.pointID;

                pointResults{id}=msg.result;
                done=done+1;

                cleanupState('pointResults')=pointResults;

                runningTime(id)=msg.result.tEnd;
                runningSteps(id)=msg.result.nSteps;

                gpuPoints(gpuPoints==id)=[];

        end

        reportProgress(false);

        if toc(saveClock)>=cfg.checkpointSeconds
            saveCheckpoint();
        end

    end


    function reportProgress(force)

        if ~force && ...
                toc(progressClock)<cfg.progressSeconds
            return;
        end

        ids=futurePoints;

        if strcmpi(cfg.executionBackend,'gpu')
            ids=gpuPoints;
        end

        if serialPoint>0
            ids=serialPoint;
        end

        fprintf( ...
            '[%6.1f min] Finished %d/%d (%.1f%%); running %d', ...
            toc(scanClock)/60,done,total, ...
            100*done/total,numel(ids));

        if ~isempty(ids)

            fprintf('; running t range %.1f--%.1f', ...
                min(runningTime(ids)),max(runningTime(ids)));

        end

        if latestPoint>0 && any(ids==latestPoint)

            [row,col]=ind2sub([ny,nx],latestPoint);

            fprintf('; latest (%.2f,%.2f): %.3g steps', ...
                cfg.gamma1(col),cfg.gamma2(row), ...
                runningSteps(latestPoint));

        end

        fprintf('\n');

        progressClock=tic;

    end


    function saveCheckpoint()

        tempFile=[checkpointFile,'.tmp.mat'];

        save(tempFile,'pointResults', ...
            'simulationCfg','-v7');

        movefile(tempFile,checkpointFile,'f');

        saveClock=tic;

    end

end


function cleanupScan(state)

try

    futures=state('futures');

    if ~isempty(futures)
        cancel(futures);
    end

catch problem

    warning('Task cancellation failed: %s', ...
        problem.message);

end

if ~state('finished')

    try

        pointResults=state('pointResults');
        simulationCfg=state('simulationCfg');

        checkpointFile=state('checkpointFile');
        tempFile=[checkpointFile,'.tmp.mat'];

        save(tempFile,'pointResults', ...
            'simulationCfg','-v7');

        movefile(tempFile,checkpointFile,'f');

        fprintf('Completed points saved.\n');

    catch problem

        warning('Checkpoint save failed: %s', ...
            problem.message);

    end

end

end


%% ============================================================
%  CPU simulation
% =============================================================

function [out,trace]=simulatePoint( ...
    g1,g2,cfg,progressSink,pointID)

if nargin<4
    progressSink=[];
end

if nargin<5
    pointID=0;
end

N=cfg.N;
n1=round(N*g1);
n2=round(N*g2);

seed=mod( ...
    cfg.seedBase+ ...
    1000003*round(1000*g1)+ ...
    9176*round(1000*g2)+ ...
    cfg.seedOffset,2147483646)+1;

out=emptyPoint();

out.n1=n1;
out.n2=n2;
out.nP=N-n1-n2;
out.seed=seed;

active=[n1>0,n2>0];

if ~any(active)

    out.status=-1;
    out.reasonCode=5;
    out.tEnd=0;
    out.nSteps=0;

    out.nRejected=0;
    out.rhsCalls=0;
    out.maxAcceptedError=0;
    out.maxRawNormError=0;

    out.solverCode=1;
    out.gpuUsed=0;

    trace=zeros(0,8);

    return;

end

sigma=initialState( ...
    N,n1,n2,cfg.perturbation,seed);

ns=round(cfg.window/cfg.sampleDt);

burnIndex=round(cfg.burnIn/cfg.sampleDt);
maxCount=round(cfg.tMax/cfg.sampleDt);

nextCheck=burnIndex+2*ns;

keepTrace=nargout>1;

ring=nan(2*ns+1,7);
ringPos=0;
ringCount=0;

if keepTrace

    traceBuffer=nan(maxCount+1,8);
    traceCount=1;

    firstObs=observe(toHost(sigma),n1,n2);

    traceBuffer(1,:)=[0,firstObs];

else

    traceBuffer=[];
    traceCount=0;
    firstObs=[];

end

if burnIndex==0

    if isempty(firstObs)
        firstObs=observe(toHost(sigma),n1,n2);
    end

    ringPos=1;
    ringCount=1;
    ring(1,:)=firstObs;

end

if cfg.burnIn>0 && ~keepTrace
    nextSample=cfg.burnIn;
else
    nextSample=cfg.sampleDt;
end

t=0;
steps=0;
rejected=0;
rhsCalls=0;
streak=0;

dtMin=inf;
dtMax=0;
maxError=0;
maxRawNormError=0;

heartbeatClock=tic;

hProposal=cfg.dtMax*cfg.stepFactor;

k1=[];
R=nan;

while t<cfg.tMax

    if isempty(k1)

        [k1,R]=rhsSphere(sigma,cfg.K);

        rhsCalls=rhsCalls+1;

    end

    freeStep=min([ ...
        hProposal, ...
        cfg.dtMax*cfg.stepFactor, ...
        0.9*cfg.phaseStep*cfg.stepFactor/ ...
        (cfg.K*max(R,1e-12)^5), ...
        0.9*cfg.feedbackStep*cfg.stepFactor/ ...
        (cfg.K*max(R,1e-12)^4*(1+R))]);

    h=min(freeStep,nextSample-t);

    clippedForSample=h<freeStep;

    assert(h>0 && t+h>t, ...
        'Integrator time step underflow.');

    [sigma,k1,R,h,hProposal,stats]= ...
        rk45Step(sigma,k1,R,h,cfg,t);

    if clippedForSample && stats.rejected==0

        hProposal=max(hProposal,freeStep);

    end

    rejected=rejected+stats.rejected;
    rhsCalls=rhsCalls+stats.rhsCalls;

    maxError=max(maxError,stats.error);
    maxRawNormError=max( ...
        maxRawNormError,stats.normError);

    t=t+h;
    steps=steps+1;

    dtMin=min(dtMin,h);
    dtMax=max(dtMax,h);

    if ~isempty(progressSink) && ...
            mod(steps,256)==0 && ...
            toc(heartbeatClock)>=cfg.progressSeconds

        msg=struct( ...
            'pointID',pointID, ...
            't',t, ...
            'steps',steps);

        if isa(progressSink,'function_handle')
            progressSink(msg);
        else
            send(progressSink,msg);
        end

        heartbeatClock=tic;

    end

    if t>=nextSample-32*eps(max(1,nextSample))

        t=nextSample;

        count=round(t/cfg.sampleDt);

        if keepTrace || count>=burnIndex

            values=observe(toHost(sigma),n1,n2);

            if keepTrace

                traceCount=traceCount+1;

                traceBuffer(traceCount,:)=[t,values];

            end

            if count>=burnIndex

                ringPos=mod(ringPos,size(ring,1))+1;

                ring(ringPos,:)=values;

                ringCount=ringCount+1;

            end

        end

        nextSample=(count+1)*cfg.sampleDt;

        if count==nextCheck

            assert(ringCount>=size(ring,1), ...
                'Two full time windows required.');

            tail=orderedTail(ring,ringPos);

            qc=windowCheck( ...
                tail,ns,active,cfg);

            if qc.meanPassed && qc.clockPassed
                streak=streak+1;
            else
                streak=0;
            end

            nextCheck=nextCheck+ns;

            if cfg.allowEarlyStop && ...
                    streak>=cfg.requiredPasses
                break;
            end

        end

    end

end

tail=orderedTail(ring,ringPos);

out=finalizePoint(out,tail,ns,streak,qc,cfg);

out.tEnd=t;
out.avgStart=t-2*cfg.window;
out.avgEnd=t;
out.avgDuration=2*cfg.window;

out.nSteps=steps;
out.dtMin=dtMin;
out.dtMax=dtMax;

out.nRejected=rejected;
out.rhsCalls=rhsCalls;

out.maxAcceptedError=maxError;
out.maxRawNormError=maxRawNormError;

out.solverCode=1;
out.gpuUsed=0;

if keepTrace
    trace=traceBuffer(1:traceCount,:);
end

end


%% ============================================================
%  GPU dynamic batch
% =============================================================

function simulateGpuBatch(pending,cfg,eventSink)

N=cfg.N;

B=min(cfg.gpuBatchSizeEffective,numel(pending));

nx=numel(cfg.gamma1);
ny=numel(cfg.gamma2);

ns=round(cfg.window/cfg.sampleDt);
burnIndex=round(cfg.burnIn/cfg.sampleDt);

capacity=2*ns+1;

sigma=zeros(N,3,B,'double','gpuArray');
k1=zeros(N,3,B,'double','gpuArray');

particleID=gpuArray((1:N)');

slotIDs=zeros(1,B);
n1=zeros(1,B);
n2=zeros(1,B);

t=zeros(1,B);
R=zeros(1,B);

hProposal=zeros(1,B);
nextSample=zeros(1,B);
nextCheck=zeros(1,B);

streak=zeros(1,B);
retryCount=zeros(1,B);

steps=zeros(1,B);
rejected=zeros(1,B);
rhsCalls=zeros(1,B);

dtMin=inf(1,B);
dtMax=zeros(1,B);
maxError=zeros(1,B);
maxNormError=zeros(1,B);

ring=cell(1,B);
pointData=cell(1,B);
lastCheck=cell(1,B);

ringPos=zeros(1,B);
ringCount=zeros(1,B);

nextPending=1;
heartbeatClock=tic;
attempts=0;

fillSlots(1:B);

while any(slotIDs>0)

    work=find(slotIDs>0);

    freeStep=min( ...
        hProposal(work),cfg.dtMax*cfg.stepFactor);

    safeR=max(R(work),1e-12);

    freeStep=min( ...
        freeStep, ...
        0.9*cfg.phaseStep*cfg.stepFactor./ ...
        (cfg.K*safeR.^5));

    freeStep=min( ...
        freeStep, ...
        0.9*cfg.feedbackStep*cfg.stepFactor./ ...
        (cfg.K*safeR.^4.*(1+R(work))));

    h=min(freeStep,nextSample(work)-t(work));

    clipped=h<freeStep;

    assert(all(h>0 & t(work)+h>t(work)), ...
        'GPU RK45 time step underflow.');

    if numel(work)==B

        s=sigma;
        derivative=k1;

    else

        s=sigma(:,:,work);
        derivative=k1(:,:,work);

    end

    [candidate,kNext,d]=rk45AttemptGpu( ...
        s,derivative,R(work),h,cfg);

    phase=cfg.K*d(2,:).^5.*h;

    feedback=cfg.K*d(2,:).^4.* ...
        (1+d(2,:)).*h;

    accepted=all(isfinite(d),1) & ...
        d(5,:)==1 & ...
        d(1,:)<=1 & ...
        phase<=cfg.phaseGuard*cfg.stepFactor* ...
        (1+1e-12) & ...
        feedback<=cfg.feedbackStep*cfg.stepFactor* ...
        (1+1e-12);

    rhsCalls(work)=rhsCalls(work)+6;

    good=find(accepted);
    lanes=work(good);

    if ~isempty(lanes)

        sigma(:,:,lanes)=candidate(:,:,good);
        k1(:,:,lanes)=kNext(:,:,good);

        R(lanes)=d(3,good);

        t(lanes)=t(lanes)+h(good);
        steps(lanes)=steps(lanes)+1;

        dtMin(lanes)=min( ...
            dtMin(lanes),h(good));

        dtMax(lanes)=max( ...
            dtMax(lanes),h(good));

        maxError(lanes)=max( ...
            maxError(lanes),d(1,good));

        maxNormError(lanes)=max( ...
            maxNormError(lanes),d(4,good));

        factor=4*ones(1,numel(good));

        nonzero=d(1,good)>0;

        factor(nonzero)=min( ...
            4,max(0.2, ...
            0.9*d(1,good(nonzero)).^(-1/5)));

        proposed=h(good).*factor;

        restore=clipped(good) & ...
            retryCount(lanes)==0;

        proposed(restore)=max( ...
            proposed(restore), ...
            freeStep(good(restore)));

        hProposal(lanes)=proposed;
        retryCount(lanes)=0;

    end

    for q=find(~accepted)

        lane=work(q);

        rejected(lane)=rejected(lane)+1;
        retryCount(lane)=retryCount(lane)+1;

        if any(~isfinite(d(:,q))) || d(5,q)~=1

            factor=0.2;

        else

            factor=min(0.9,max(0.1, ...
                0.9*max(d(1,q),1e-16)^(-1/5)));

            if phase(q)>cfg.phaseGuard*cfg.stepFactor

                factor=min( ...
                    factor, ...
                    0.9*cfg.phaseGuard* ...
                    cfg.stepFactor/phase(q));

            end

            if feedback(q)> ...
                    cfg.feedbackStep*cfg.stepFactor

                factor=min( ...
                    factor, ...
                    0.9*cfg.feedbackStep* ...
                    cfg.stepFactor/feedback(q));

            end

        end

        hProposal(lane)=h(q)*factor;

        if retryCount(lane)>=60

            [row,col]=ind2sub( ...
                [ny,nx],slotIDs(lane));

            error( ...
                'GPU RK45 failed at gamma=(%g,%g).', ...
                cfg.gamma1(col),cfg.gamma2(row));

        end

    end

    reached=lanes( ...
        t(lanes)>=nextSample(lanes)- ...
        32*eps(max(1,nextSample(lanes))));

    finished=[];

    if ~isempty(reached)

        t(reached)=nextSample(reached);

        values=observeGpu( ...
            sigma(:,:,reached), ...
            n1(reached),n2(reached),particleID);

        for q=1:numel(reached)

            lane=reached(q);

            count=round(t(lane)/cfg.sampleDt);

            ringPos(lane)=mod( ...
                ringPos(lane),capacity)+1;

            ring{lane}(ringPos(lane),:)= ...
                values(:,q)';

            ringCount(lane)=ringCount(lane)+1;

            nextSample(lane)= ...
                (count+1)*cfg.sampleDt;

            if count==nextCheck(lane)

                assert(ringCount(lane)>=capacity, ...
                    'GPU: insufficient time window.');

                tail=orderedTail( ...
                    ring{lane},ringPos(lane));

                qc=windowCheck( ...
                    tail,ns, ...
                    [n1(lane)>0,n2(lane)>0],cfg);

                lastCheck{lane}=qc;

                if qc.meanPassed && qc.clockPassed
                    streak(lane)=streak(lane)+1;
                else
                    streak(lane)=0;
                end

                nextCheck(lane)=nextCheck(lane)+ns;

            end

            if (cfg.allowEarlyStop && ...
                    streak(lane)>=cfg.requiredPasses) || ...
                    t(lane)>=cfg.tMax

                tail=orderedTail( ...
                    ring{lane},ringPos(lane));

                out=finalizePoint( ...
                    pointData{lane},tail,ns, ...
                    streak(lane),lastCheck{lane},cfg);

                out.tEnd=t(lane);

                out.avgStart=t(lane)-2*cfg.window;
                out.avgEnd=t(lane);
                out.avgDuration=2*cfg.window;

                out.nSteps=steps(lane);
                out.dtMin=dtMin(lane);
                out.dtMax=dtMax(lane);

                out.nRejected=rejected(lane);
                out.rhsCalls=rhsCalls(lane);

                out.maxAcceptedError=maxError(lane);
                out.maxRawNormError=maxNormError(lane);

                out.solverCode=1;
                out.gpuUsed=1;

                eventSink(struct( ...
                    'kind','complete', ...
                    'pointID',slotIDs(lane), ...
                    'result',out));

                slotIDs(lane)=0;

                ring{lane}=[];
                pointData{lane}=[];
                lastCheck{lane}=[];

                finished(end+1)=lane; %#ok<AGROW>

            end

        end

    end

    if ~isempty(finished)
        fillSlots(finished);
    end

    attempts=attempts+1;

    if mod(attempts,128)==0 && ...
            toc(heartbeatClock)>=cfg.progressSeconds

        sendHeartbeat();

    end

end


    function fillSlots(slots)

        filled=[];
        host=zeros(N,3,numel(slots));

        for lane=slots

            while nextPending<=numel(pending)

                id=pending(nextPending);
                nextPending=nextPending+1;

                [row,col]=ind2sub([ny,nx],id);

                g1=cfg.gamma1(col);
                g2=cfg.gamma2(row);

                a=round(N*g1);
                b=round(N*g2);

                if a+b==0

                    out=simulatePoint(g1,g2,cfg);

                    eventSink(struct( ...
                        'kind','complete', ...
                        'pointID',id, ...
                        'result',out));

                    continue;

                end

                seed=mod( ...
                    cfg.seedBase+ ...
                    1000003*round(1000*g1)+ ...
                    9176*round(1000*g2)+ ...
                    cfg.seedOffset,2147483646)+1;

                out=emptyPoint();

                out.n1=a;
                out.n2=b;
                out.nP=N-a-b;
                out.seed=seed;

                pointData{lane}=out;

                slotIDs(lane)=id;
                n1(lane)=a;
                n2(lane)=b;

                t(lane)=0;

                hProposal(lane)= ...
                    cfg.dtMax*cfg.stepFactor;

                if cfg.burnIn>0
                    nextSample(lane)=cfg.burnIn;
                else
                    nextSample(lane)=cfg.sampleDt;
                end

                nextCheck(lane)=burnIndex+2*ns;

                streak(lane)=0;
                retryCount(lane)=0;

                steps(lane)=0;
                rejected(lane)=0;
                rhsCalls(lane)=1;

                dtMin(lane)=inf;
                dtMax(lane)=0;
                maxError(lane)=0;
                maxNormError(lane)=0;

                ring{lane}=nan(capacity,7);

                ringPos(lane)=0;
                ringCount(lane)=0;

                lastCheck{lane}=[];

                filled(end+1)=lane; %#ok<AGROW>

                host(:,:,numel(filled))= ...
                    initialState( ...
                    N,a,b,cfg.perturbation,seed);

                break;

            end

        end

        if ~isempty(filled)

            initial=gpuArray( ...
                host(:,:,1:numel(filled)));

            sigma(:,:,filled)=initial;

            [derivative,initialR]= ...
                rhsGpu(initial,cfg.K);

            k1(:,:,filled)=derivative;

            R(filled)=gather( ...
                reshape(initialR,1,[]));

            if burnIndex==0

                values=observeGpu( ...
                    initial,n1(filled), ...
                    n2(filled),particleID);

                for q=1:numel(filled)

                    lane=filled(q);

                    ring{lane}(1,:)=values(:,q)';

                    ringPos(lane)=1;
                    ringCount(lane)=1;

                end

            end

        end

        eventSink(struct( ...
            'kind','active', ...
            'pointIDs',slotIDs(slotIDs>0)));

        if toc(heartbeatClock)>=cfg.progressSeconds
            sendHeartbeat();
        end

    end


    function sendHeartbeat()

        active=find(slotIDs>0);

        eventSink(struct( ...
            'kind','progress', ...
            'pointIDs',slotIDs(active), ...
            't',t(active), ...
            'steps',steps(active)));

        heartbeatClock=tic;

    end

end


%% ============================================================
%  Finalize statistics
% =============================================================

function out=finalizePoint( ...
    out,tail,ns,streak,qc,cfg)

active=[out.n1>0,out.n2>0];

[mu,sd]=timeStatistics(tail);

out.Z1mean=mu(1);
out.Z2mean=mu(2);

out.Z1std=sd(1);
out.Z2std=sd(2);

out.Z1end=tail(end,1);
out.Z2end=tail(end,2);

out.Rmean=mu(3);
out.Pmean=mu(4);
out.betaMean=mu(5);

out.polarRMS1=mu(6);
out.polarRMS2=mu(7);

if all(isfinite(tail(:,4)))

    out.Pmin=min(tail(:,4));

    out.PspreadFraction= ...
        trapz(double( ...
        tail(:,4)<cfg.pCoherenceThreshold))/(2*ns);

end

out.status=double(streak>=cfg.requiredPasses);

out.reasonCode=checkReason( ...
    qc,streak,cfg.requiredPasses);

out.meanCheckPassed=double(qc.meanPassed);
out.clockCheckPassed=double(qc.clockPassed);

out.consecutivePasses=streak;

out.windowMeanDrift1=qc.drift(1);
out.windowMeanDrift2=qc.drift(2);

out.windowMeanDrift=qc.maxDrift;

out.windowClock1=qc.clock(1);
out.windowClock2=qc.clock(2);

out.minWindowClock=min(qc.clock);

if qc.valid

    zStd=sd(1:2);

    out.hasFluctuations=double( ...
        any(zStd(active)>cfg.fluctuationTol));

end

end


function tail=orderedTail(ring,last)

ids=mod( ...
    last+(0:size(ring,1)-1),size(ring,1))+1;

tail=ring(ids,:);

end


function value=toHost(value)

if isa(value,'gpuArray')
    value=gather(value);
end

end


%% ============================================================
%  Dynamics and RK45
% =============================================================

function [v,R]=rhsSphere(sigma,K)

sigma=normalizeRows(sigma);

rho=mean(sigma,1);

R2=sum(rho.*rho);

a=sum(sigma.*rho,2);
a2=a.*a;

v=K*((16*a2-12*R2).*a2+R2*R2).* ...
    (rho-a.*sigma);

R=sqrt(R2);

end


function [next,kNext,RNext,h,hNext,stats]= ...
    rk45Step(sigma,k1,R,h,cfg,currentTime)

stats.rejected=0;
stats.rhsCalls=0;

for attempt=1:60

    assert(h>0 && currentTime+h>currentTime, ...
        'RK45 step underflow.');

    [k2,r2]=rhsSphere( ...
        sigma+h*(k1/5),cfg.K);

    [k3,r3]=rhsSphere( ...
        sigma+h*(3*k1/40+9*k2/40),cfg.K);

    [k4,r4]=rhsSphere( ...
        sigma+h*(44*k1/45-56*k2/15+ ...
        32*k3/9),cfg.K);

    [k5,r5]=rhsSphere( ...
        sigma+h*(19372*k1/6561- ...
        25360*k2/2187+ ...
        64448*k3/6561-212*k4/729),cfg.K);

    [k6,r6]=rhsSphere( ...
        sigma+h*(9017*k1/3168- ...
        355*k2/33+ ...
        46732*k3/5247+49*k4/176- ...
        5103*k5/18656),cfg.K);

    raw5=sigma+h*( ...
        35*k1/384+ ...
        500*k3/1113+ ...
        125*k4/192- ...
        2187*k5/6784+ ...
        11*k6/84);

    next=normalizeRows(raw5);

    [k7,r7]=rhsSphere(next,cfg.K);

    raw4=sigma+h*( ...
        5179*k1/57600+ ...
        7571*k3/16695+ ...
        393*k4/640- ...
        92097*k5/339200+ ...
        187*k6/2100+ ...
        k7/40);

    scale=cfg.absTol+ ...
        cfg.relTol*max(abs(sigma),abs(next));

    err=max(abs(raw5-raw4)./scale,[],'all');

    err=max( ...
        err,max(abs(next-raw5)./scale,[],'all'));

    peakR=max([R,r2,r3,r4,r5,r6,r7]);

    normError=max(abs(sum(raw5.*raw5,2)-1));

    finiteState= ...
        all(isfinite(next),'all') & ...
        all(isfinite(raw4),'all');

    diagnostics=toHost([ ...
        err,peakR,r7,normError, ...
        double(finiteState)]);

    err=diagnostics(1);
    peakR=diagnostics(2);

    effectiveStep=cfg.K*peakR^5*h;

    feedbackStep= ...
        cfg.K*peakR^4*(1+peakR)*h;

    stats.rhsCalls=stats.rhsCalls+6;

    if isfinite(err) && ...
            all(isfinite(diagnostics)) && ...
            diagnostics(5)==1 && ...
            err<=1 && ...
            effectiveStep<= ...
            cfg.phaseGuard*cfg.stepFactor* ...
            (1+1e-12) && ...
            feedbackStep<= ...
            cfg.feedbackStep*cfg.stepFactor* ...
            (1+1e-12)

        kNext=k7;
        RNext=diagnostics(3);

        if err==0
            factor=4;
        else
            factor=min(4,max(0.2, ...
                0.9*err^(-1/5)));
        end

        hNext=h*factor;

        stats.error=err;
        stats.normError=diagnostics(4);

        return;

    end

    stats.rejected=stats.rejected+1;

    if ~isfinite(err) || ...
            any(~isfinite(diagnostics)) || ...
            diagnostics(5)~=1

        factor=0.2;

    else

        factor=min(0.9,max(0.1, ...
            0.9*max(err,1e-16)^(-1/5)));

        if effectiveStep> ...
                cfg.phaseGuard*cfg.stepFactor

            factor=min( ...
                factor, ...
                0.9*cfg.phaseGuard* ...
                cfg.stepFactor/effectiveStep);

        end

        if feedbackStep> ...
                cfg.feedbackStep*cfg.stepFactor

            factor=min( ...
                factor, ...
                0.9*cfg.feedbackStep* ...
                cfg.stepFactor/feedbackStep);

        end

    end

    h=h*factor;

    assert(h>0,'RK45 step underflow.');

end

error('RK45 failed after 60 attempts.');

end


%% ============================================================
%  GPU RK45
% =============================================================

function [next,kNext,d]=rk45AttemptGpu( ...
    s,k1,R,h,cfg)

B=size(s,3);

hg=gpuArray(reshape(h,1,1,B));

[k2,r2]=rhsGpu( ...
    gpuCombination( ...
    s,hg,k1,0,0,0,0,0, ...
    1/5,0,0,0,0,0),cfg.K);

[k3,r3]=rhsGpu( ...
    gpuCombination( ...
    s,hg,k1,k2,0,0,0,0, ...
    3/40,9/40,0,0,0,0),cfg.K);

[k4,r4]=rhsGpu( ...
    gpuCombination( ...
    s,hg,k1,k2,k3,0,0,0, ...
    44/45,-56/15,32/9,0,0,0),cfg.K);

[k5,r5]=rhsGpu( ...
    gpuCombination( ...
    s,hg,k1,k2,k3,k4,0,0, ...
    19372/6561,-25360/2187, ...
    64448/6561,-212/729,0,0),cfg.K);

[k6,r6]=rhsGpu( ...
    gpuCombination( ...
    s,hg,k1,k2,k3,k4,k5,0, ...
    9017/3168,-355/33, ...
    46732/5247,49/176, ...
    -5103/18656,0),cfg.K);

raw5=gpuCombination( ...
    s,hg,k1,k3,k4,k5,k6,0, ...
    35/384,500/1113,125/192, ...
    -2187/6784,11/84,0);

[next,normError]=normalizeGpuPages(raw5);

[kNext,r7]=rhsGpu(next,cfg.K);

raw4=gpuCombination( ...
    s,hg,k1,k3,k4,k5,k6,kNext, ...
    5179/57600,7571/16695, ...
    393/640,-92097/339200, ...
    187/2100,1/40);

[componentError,finiteComponent]=arrayfun( ...
    @gpuEmbeddedError, ...
    s,raw5,raw4,next, ...
    cfg.absTol,cfg.relTol);

err=max(max(componentError,[],1),[],2);

finiteState=all(all(finiteComponent,1),2);

peakR=max(cat(1, ...
    gpuArray(reshape(R,1,1,B)), ...
    r2,r3,r4,r5,r6,r7),[],1);

d=gather([ ...
    reshape(err,1,B); ...
    reshape(peakR,1,B); ...
    reshape(r7,1,B); ...
    reshape(normError,1,B); ...
    double(reshape(finiteState,1,B))]);

end


function y=gpuCombination( ...
    s,h,k1,k2,k3,k4,k5,k6, ...
    b1,b2,b3,b4,b5,b6)

y=arrayfun( ...
    @gpuCombinationElement, ...
    s,h,k1,k2,k3,k4,k5,k6, ...
    b1,b2,b3,b4,b5,b6);

end


function y=gpuCombinationElement( ...
    s,h,k1,k2,k3,k4,k5,k6, ...
    b1,b2,b3,b4,b5,b6)

y=s+h*( ...
    b1*k1+b2*k2+b3*k3+ ...
    b4*k4+b5*k5+b6*k6);

end


function [v,R]=rhsGpu(s,K)

s=normalizeGpuPages(s);

N=size(s,1);
B=size(s,3);

rho=mean(s,1);

R2=sum(rho.*rho,2);
R=sqrt(R2);

[vx,vy,vz]=arrayfun( ...
    @gpuForceElement, ...
    reshape(s(:,1,:),N,B), ...
    reshape(s(:,2,:),N,B), ...
    reshape(s(:,3,:),N,B), ...
    reshape(rho(1,1,:),1,B), ...
    reshape(rho(1,2,:),1,B), ...
    reshape(rho(1,3,:),1,B), ...
    reshape(R2,1,B),K);

v=cat(2, ...
    reshape(vx,N,1,B), ...
    reshape(vy,N,1,B), ...
    reshape(vz,N,1,B));

end


function [vx,vy,vz]=gpuForceElement( ...
    x,y,z,rx,ry,rz,R2,K)

a=x*rx+y*ry+z*rz;
a2=a*a;

factor=K*((16*a2-12*R2)*a2+R2*R2);

vx=factor*(rx-a*x);
vy=factor*(ry-a*y);
vz=factor*(rz-a*z);

end


function [s,normError]=normalizeGpuPages(s)

N=size(s,1);
B=size(s,3);

x=reshape(s(:,1,:),N,B);
y=reshape(s(:,2,:),N,B);
z=reshape(s(:,3,:),N,B);

if nargout>1

    [x,y,z,e]=arrayfun( ...
        @gpuNormalizeElement,x,y,z);

    normError=reshape(max(e,[],1),1,1,B);

else

    [x,y,z]=arrayfun( ...
        @gpuNormalizeElement,x,y,z);

end

s=cat(2, ...
    reshape(x,N,1,B), ...
    reshape(y,N,1,B), ...
    reshape(z,N,1,B));

end


function [x,y,z,e]=gpuNormalizeElement(x,y,z)

r2=x*x+y*y+z*z;
r=sqrt(r2);

e=abs(r2-1);

x=x/r;
y=y/r;
z=z/r;

end


function [err,valid]=gpuEmbeddedError( ...
    s,raw5,raw4,next,absTol,relTol)

scale=absTol+ ...
    relTol*max(abs(s),abs(next));

err=max( ...
    abs(raw5-raw4), ...
    abs(next-raw5))/scale;

valid=isfinite(next) && ...
    isfinite(raw4) && ...
    isfinite(raw5) && ...
    isfinite(err);

end


%% ============================================================
%  GPU observations
%  Important: azimuthal axis is rho, NOT P
% =============================================================

function values=observeGpu(s,n1,n2,particleID)

N=size(s,1);
B=size(s,3);

n1g=gpuArray(reshape(n1,1,B));
n2g=gpuArray(reshape(n2,1,B));

maskP=particleID>reshape( ...
    n1g+n2g,1,1,B);

pMean=sum(s.*maskP,1)./ ...
    reshape(N-n1g-n2g,1,1,B);

pNorm=sqrt(sum(pMean.*pMean,2));

p=pMean./max(pNorm,1e-12);

rho=mean(s,1);
R=sqrt(sum(rho.*rho,2));

% P direction for polar diagnostics only

px=reshape(p(1,1,:),1,B);
py=reshape(p(1,2,:),1,B);
pz=reshape(p(1,3,:),1,B);

% Instantaneous rho reference axis

zAxis=rho./max(R,1e-12);

zx=reshape(zAxis(1,1,:),1,B);
zy=reshape(zAxis(1,2,:),1,B);
zz=reshape(zAxis(1,3,:),1,B);

[e1x,e1y,e1z,e2x,e2y,e2z]= ...
    arrayfun(@gpuAzimuthBasis,zx,zy,zz);

% Azimuth relative to rho;
% polar diagnostic relative to P.

[c1,s1,polar1,bad1,c2,s2,polar2,bad2]= ...
    arrayfun(@gpuAzimuthElement, ...
    reshape(s(:,1,:),N,B), ...
    reshape(s(:,2,:),N,B), ...
    reshape(s(:,3,:),N,B), ...
    px,py,pz, ...
    e1x,e1y,e1z, ...
    e2x,e2y,e2z, ...
    particleID,n1g,n2g);

z1=hypot(sum(c1,1),sum(s1,1))./n1g;
z2=hypot(sum(c2,1),sum(s2,1))./n2g;

polar1=sqrt(sum(polar1,1)./n1g);
polar2=sqrt(sum(polar2,1)./n2g);

beta=acos(max(-1,min(1, ...
    sum(rho.*p,2)./max(R,1e-15))));

raw=gather([ ...
    z1;z2; ...
    reshape(R,1,B); ...
    reshape(pNorm,1,B); ...
    reshape(beta,1,B); ...
    polar1;polar2; ...
    sum(bad1,1);sum(bad2,1)]);

values=raw(1:7,:);

values(1,n1==0 | raw(8,:)>0)=nan;
values(6,n1==0)=nan;

values(2,n2==0 | raw(9,:)>0)=nan;
values(7,n2==0)=nan;

values(5, ...
    ~isfinite(values(3,:)) | ...
    values(3,:)<=1e-15)=nan;

undefinedP= ...
    ~isfinite(values(4,:)) | ...
    values(4,:)<=1e-12;

values([1,2,5,6,7],undefinedP)=nan;

% Rho axis is undefined if |rho| is zero.

undefinedR= ...
    ~isfinite(values(3,:)) | ...
    values(3,:)<=1e-12;

values(1:2,undefinedR)=nan;

end


function [a,b,c,d,e,f]=gpuAzimuthBasis( ...
    zx,zy,zz)

if abs(zz)<0.9

    r=sqrt(zx*zx+zy*zy);

    a=-zy/r;
    b=zx/r;
    c=0;

else

    r=sqrt(zy*zy+zz*zz);

    a=0;
    b=-zz/r;
    c=zy/r;

end

d=zy*c-zz*b;
e=zz*a-zx*c;
f=zx*b-zy*a;

end


function [c1,s1,polar1,bad1,c2,s2,polar2,bad2]= ...
    gpuAzimuthElement( ...
    x,y,z,px,py,pz, ...
    e1x,e1y,e1z,e2x,e2y,e2z,id,n1,n2)

c1=0;
s1=0;
polar1=0;
bad1=0;

c2=0;
s2=0;
polar2=0;
bad2=0;

if id<=n1+n2

    % Azimuth relative to rho axis

    a=x*e1x+y*e1y+z*e1z;
    b=x*e2x+y*e2y+z*e2z;

    r=hypot(a,b);

    bad=0;
    cosine=0;
    sine=0;

    if ~isfinite(r) || r<=1e-10

        bad=1;

    else

        cosine=a/r;
        sine=b/r;

    end

    % Polar deviation still relative to P

    theta=acos(max(-1,min(1, ...
        x*px+y*py+z*pz)));

    if id<=n1

        c1=cosine;
        s1=sine;
        bad1=bad;

        polar1=(theta-2*pi/5)^2;

    else

        c2=cosine;
        s2=sine;
        bad2=bad;

        polar2=(theta-4*pi/5)^2;

    end

end

end


%% ============================================================
%  Initial state and CPU observation
% =============================================================

function out=emptyPoint()

names={ ...
    'Z1mean','Z2mean', ...
    'Z1std','Z2std', ...
    'Z1end','Z2end', ...
    'Rmean','Pmean', ...
    'Pmin','PspreadFraction', ...
    'betaMean', ...
    'polarRMS1','polarRMS2', ...
    'tEnd','status','reasonCode', ...
    'meanCheckPassed','clockCheckPassed', ...
    'consecutivePasses', ...
    'windowMeanDrift1','windowMeanDrift2', ...
    'windowMeanDrift', ...
    'windowClock1','windowClock2', ...
    'minWindowClock', ...
    'hasFluctuations', ...
    'avgStart','avgEnd','avgDuration', ...
    'nSteps','dtMin','dtMax', ...
    'n1','n2','nP','seed', ...
    'nRejected','rhsCalls', ...
    'maxAcceptedError','maxRawNormError', ...
    'solverCode','gpuUsed'};

out=struct();

for k=1:numel(names)
    out.(names{k})=nan;
end

end


function sigma=initialState( ...
    N,n1,n2,epsilon,seed)

sigma=zeros(N,3);
state=seed;

for k=1:N

    if k<=n1

        theta=2*pi/5;

    elseif k<=n1+n2

        theta=4*pi/5;

    else

        theta=0;

    end

    state=mod(16807*state,2147483647);

    az=2*pi*state/2147483647;

    v=[ ...
        sin(theta)*cos(az), ...
        sin(theta)*sin(az), ...
        cos(theta)];

    for d=1:3

        state=mod(16807*state,2147483647);

        v(d)=v(d)+ ...
            epsilon*(2*state/2147483647-1);

    end

    sigma(k,:)=v/norm(v);

end

end


function a=normalizeRows(a)

a=a./sqrt(sum(a.^2,2));

end


function values=observe(sigma,n1,n2)

% Mean direction of the P population

pMean=mean( ...
    sigma(n1+n2+1:end,:),1);

pNorm=norm(pMean);

% Instantaneous global mean field

rho=mean(sigma,1);
R=norm(rho);

if ~isfinite(pNorm) || pNorm<=1e-12

    values=[nan,nan,R,pNorm,nan,nan,nan];

    return;

end

p=pMean/pNorm;

% Azimuthal reference axis = rho / |rho|

if ~isfinite(R) || R<=1e-12

    values=[nan,nan,R,pNorm,nan,nan,nan];

    return;

end

zAxis=rho/R;

if abs(zAxis(3))<0.9
    reference=[0,0,1];
else
    reference=[1,0,0];
end

e1=cross(reference,zAxis);
e1=e1/norm(e1);

e2=cross(zAxis,e1);

Z=nan(1,2);
polar=nan(1,2);

for group=1:2

    if group==1

        ids=1:n1;

    else

        ids=n1+(1:n2);

    end

    if isempty(ids)
        continue;
    end

    s=sigma(ids,:);

    % Projection onto the plane perpendicular to rho

    x=s*e1';
    y=s*e2';

    r=hypot(x,y);

    if all(r>1e-10)

        Z(group)=abs(mean( ...
            (x+1i*y)./r));

    end

    % Keep the polar diagnostic relative to P

    theta=acos(max(-1,min(1,s*p')));

    polar(group)=sqrt(mean( ...
        (theta-group*2*pi/5).^2));

end

beta=acos(max(-1,min(1, ...
    dot(rho,p)/max(R,1e-15))));

if R<=1e-15
    beta=nan;
end

values=[Z,R,pNorm,beta,polar];

end


%% ============================================================
%  Time-window statistics and convergence
% =============================================================

function qc=windowCheck(tail,ns,active,cfg)

assert(size(tail,1)==2*ns+1, ...
    'Two complete windows are required.');

mu=nan(2,7);
qc.clock=nan(1,2);

for b=1:2

    block=tail((b-1)*ns+(1:ns+1),:);

    mu(b,:)=timeStatistics(block);

    qc.clock(b)= ...
        cfg.K*cfg.sampleDt* ...
        trapz(block(:,3).^5);

end

qc.drift=abs( ...
    mu(2,1:2)-mu(1,1:2));

qc.drift(~active)=nan;

activeIds=find(active);

qc.valid=~isempty(activeIds) && ...
    all(all(isfinite(tail(:,activeIds))));

qc.meanPassed=qc.valid && ...
    all(qc.drift(active)<=cfg.meanTol);

qc.maxDrift=nan;

if qc.valid
    qc.maxDrift=max(qc.drift(active));
end

qc.clockPassed= ...
    all(isfinite(qc.clock)) && ...
    all(tail(:,3)>=0) && ...
    all(qc.clock>=cfg.minWindowClock);

end


function [mu,sd]=timeStatistics(values)

intervals=size(values,1)-1;

mu=trapz(values,1)/intervals;

if nargout>1

    sd=sqrt( ...
        trapz((values-mu).^2,1)/intervals);

end

end


function reason=checkReason( ...
    qc,streak,requiredPasses)

if ~qc.valid

    reason=4;

elseif ~qc.meanPassed

    reason=1;

elseif ~qc.clockPassed

    reason=2;

elseif streak<requiredPasses

    reason=3;

else

    reason=0;

end

end


function results=packResults(points,cfg)

names=fieldnames(points{1});

results.gamma1=cfg.gamma1;
results.gamma2=cfg.gamma2;

for k=1:numel(names)

    key=names{k};

    grid=nan(size(points));

    for j=1:numel(points)
        grid(j)=points{j}.(key);
    end

    results.(key)=grid;

end

results.statusCodes=[-1,0,1];

results.statusLabels={ ...
    'not applicable: both rings absent', ...
    'unresolved', ...
    'mean converged'};

results.reasonLabels={ ...
    'passed', ...
    'mean drift too large', ...
    'insufficient effective time', ...
    'need more consecutive passes', ...
    'undefined active-ring azimuth', ...
    'both rings absent'};

results.meanTolerance=cfg.meanTol;
results.requiredWindowClock=cfg.minWindowClock;

results.checkVersion=cfg.checkVersion;
results.engineVersion=cfg.engineVersion;

results.integrator=cfg.integrator;
results.backend=cfg.backend;

if isfield(cfg,'executionBackend')
    results.executionBackend=cfg.executionBackend;
end

if isfield(cfg,'gpuBatchSizeEffective')
    results.gpuBatchSize=cfg.gpuBatchSizeEffective;
end

results.averageDefinition= ...
    'Time average of abs(Z) with instantaneous rho axis';

end


%% ============================================================
%  Phase diagrams
% =============================================================

function drawMap(results,cfg,group)

% The color field contains only measured time-averaged |Z|.
field=sprintf('Z%dmean',group);
values=results.(field);

f=figure( ...
    'Color','w', ...
    'Position',[80+40*group,60,900,790]);

ax=axes(f, ...
    'Position',[.135,.24,.67,.66]);

h=imagesc( ...
    ax,results.gamma1, ...
    results.gamma2,values,[0,1]);

set(h,'AlphaData',isfinite(values));
set(ax,'Color',[.88,.88,.88]);

set(ax, ...
    'YDir','normal', ...
    'XLim',[0,.5], ...
    'YLim',[0,.35], ...
    'XTick',0:.1:.5, ...
    'YTick',0:.05:.35, ...
    'FontName',cfg.fontName, ...
    'FontSize',cfg.tickFont, ...
    'LineWidth',1.3, ...
    'TickDir','out', ...
    'Layer','top');

xlabel(ax,'$\gamma_{C_1}$', ...
    'Interpreter','latex', ...
    'FontSize',cfg.labelFont);

ylabel(ax,'$\gamma_{C_2}$', ...
    'Interpreter','latex', ...
    'FontSize',cfg.labelFont);

title(ax, ...
    sprintf('Ring C%d',group), ...
    'FontSize',cfg.labelFont, ...
    'FontWeight','normal');

colormap(ax,sequentialMap());

cb=colorbar(ax);
cb.Ticks=0:.2:1;
cb.FontSize=cfg.tickFont;
cb.Label.Interpreter='latex';
cb.Label.String=sprintf( ...
    '$\langle |Z_{C_%d}|\rangle_t$',group);
cb.Label.FontSize=cfg.labelFont;

ax.Position=[.135,.24,.64,.66];
cb.Position=[.82,.24,.025,.66];

if cfg.saveFigures
    savePlot(f,fullfile(cfg.outputDir, ...
        sprintf('C%d_simulation',group)));
end

end


function map=sequentialMap()

anchors=[ ...
    .267,.005,.329; ...
    .283,.141,.458; ...
    .254,.265,.530; ...
    .207,.372,.553; ...
    .164,.471,.558; ...
    .128,.567,.551; ...
    .135,.659,.518; ...
    .267,.749,.441; ...
    .478,.821,.318; ...
    .741,.873,.150; ...
    .993,.906,.144];

map=interp1( ...
    linspace(0,1,size(anchors,1)), ...
    anchors,linspace(0,1,256),'linear');

end


%% ============================================================
%  Diagnostics
% =============================================================

function drawDiagnostics(results,cfg)

f=figure( ...
    'Color','w', ...
    'Position',[100,80,1660,900]);

fields={ ...
    'windowMeanDrift1', ...
    'windowMeanDrift2', ...
    'status', ...
    'minWindowClock', ...
    'tEnd', ...
    'reasonCode'};

labels={ ...
    sprintf('C1 mean drift (tol=%.3g)', ...
    results.meanTolerance), ...
    sprintf('C2 mean drift (tol=%.3g)', ...
    results.meanTolerance), ...
    'Mean check status', ...
    sprintf('Minimum window clock (target %.3g)', ...
    results.requiredWindowClock), ...
    'End time', ...
    'Check detail'};

for k=1:6

    ax=subplot(2,3,k);

    cb=diagnosticPanel( ...
        ax,results,fields{k},labels{k},cfg);

    if k==3

        colormap(ax,[ ...
            .75,.75,.75; ...
            .84,.33,.22; ...
            .16,.63,.40]);

        caxis(ax,[-1.5,1.5]);

        cb.Ticks=[-1,0,1];

        cb.TickLabels={ ...
            'N/A','Unresolved','Passed'};

    elseif k==6

        colormap(ax,[ ...
            .16,.63,.40; ...
            .84,.33,.22; ...
            .73,.50,.17; ...
            .33,.56,.79; ...
            .57,.36,.69; ...
            .75,.75,.75]);

        caxis(ax,[-.5,5.5]);

        cb.Ticks=0:5;

        cb.TickLabels={ ...
            'Passed','Mean drift','Slow field', ...
            'More windows','Undefined Z','No rings'};

    end

end

if cfg.saveFigures

    savePlot(f,fullfile(cfg.outputDir, ...
        'convergence_diagnostics'));

end

f=figure( ...
    'Color','w', ...
    'Position',[120,100,1660,900]);

fields={ ...
    'Z1std','Z2std', ...
    'Pmean','Pmin', ...
    'Rmean','hasFluctuations'};

labels={ ...
    'Temporal std of |Z_1|', ...
    'Temporal std of |Z_2|', ...
    'Mean P coherence', ...
    'Minimum P coherence', ...
    'Mean R', ...
    'Fluctuations (diagnostic only)'};

for k=1:6

    ax=subplot(2,3,k);

    cb=diagnosticPanel( ...
        ax,results,fields{k},labels{k},cfg);

    if k==3 || k==4 || k==5
        caxis(ax,[0,1]);
    end

    if k==6

        colormap(ax,[ ...
            .88,.88,.88; ...
            .20,.54,.75]);

        caxis(ax,[-.5,1.5]);

        cb.Ticks=[0,1];

        cb.TickLabels={'Small','Present'};

    end

end

if cfg.saveFigures

    savePlot(f,fullfile(cfg.outputDir, ...
        'fluctuation_P_diagnostics'));

end

end


function cb=diagnosticPanel( ...
    ax,results,field,label,cfg)

values=results.(field);

h=imagesc( ...
    ax,results.gamma1,results.gamma2,values);

set(h,'AlphaData',isfinite(values));

axis(ax,'xy');

xlim(ax,[0,.5]);
ylim(ax,[0,.35]);

set(ax, ...
    'FontName',cfg.fontName, ...
    'FontSize',16, ...
    'Color',[.88,.88,.88], ...
    'LineWidth',1);

xlabel(ax,'\gamma_{C1}');
ylabel(ax,'\gamma_{C2}');

title(ax,label,'FontSize',16);

cb=colorbar(ax);
colormap(ax,sequentialMap());

end


function writeSummary(results,cfg)

[g1,g2]=meshgrid( ...
    results.gamma1,results.gamma2);

summary=table( ...
    g1(:),g2(:), ...
    'VariableNames', ...
    {'gamma_C1','gamma_C2'});

names=fieldnames(emptyPoint());

for k=1:numel(names)

    values=results.(names{k});

    summary.(names{k})=values(:);

end

writetable( ...
    summary, ...
    fullfile(cfg.outputDir, ...
    'phase_meancheck_summary.csv'));

end


%% ============================================================
%  Save figures
% =============================================================

function savePlot(f,base)

savefig(f,[base,'.fig']);

if exist('exportgraphics','file')

    exportgraphics( ...
        f,[base,'.png'],'Resolution',300);

    exportgraphics( ...
        f,[base,'.pdf'],'ContentType','vector');

else

    set(f,'PaperPositionMode','auto');

    print(f,[base,'.png'],'-dpng','-r300');

    print(f,[base,'.pdf'],'-dpdf','-painters');

end

end
