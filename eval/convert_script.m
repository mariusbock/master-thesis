function convert_script
date = '20200706/';
timestamp = '114353/';

data_dir = '../data/MOT/MOT17/';    
save_dir =  ['../logs/' date timestamp];
train_list = {
    'MOT17-09-DPM' ...
    'MOT17-09-FRCNN' ...
    'MOT17-09-SDP' ...
};

eval_types = {
    'val' ...
    'test'
};

pred_types = {
    'full'
};

test_list = {
%     'MOT17-01-SDP' ...
%     'MOT17-03-SDP' ...
%     'MOT17-06-SDP' ...
%     'MOT17-07-SDP' ...
%     'MOT17-08-SDP' ...
%     'MOT17-12-SDP' ...
%     'MOT17-14-SDP' ...
%     'MOT17-01-DPM' ...
%     'MOT17-03-DPM' ...
%     'MOT17-06-DPM' ...
%     'MOT17-07-DPM' ...
%     'MOT17-08-DPM' ...
%     'MOT17-12-DPM' ...
%     'MOT17-14-DPM' ...
%     'MOT17-01-FRCNN' ...
%     'MOT17-03-FRCNN' ...
%     'MOT17-06-FRCNN' ...
%     'MOT17-07-FRCNN' ...
%     'MOT17-08-FRCNN' ...
%     'MOT17-12-FRCNN' ...
%     'MOT17-14-FRCNN'
    };

mota=[];
minscore =  [0 0 0 0 0 0 0  -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 0 0 0 0 0 0 0];
minlscore = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
minlscore(8:14)=0.2;
minlscore(1:7)=0.5;
minlscore(15:21)=0.5;

is_smooth = [1 1 1 1 1 1 1 1 1 ] ;
fillin = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ];
% shortest trajectory of pedestrian that is accepted
ml = [5 5 5 5 5 5 5 ...
      5 5 5 5 5 5 5 ...
      5 5 5 5 5 5 5 ];

for i = 1:length(train_list)
    for j = 1:length(eval_types)
        for k = 1:length(pred_types)
            train_sequence = train_list{i};
            %tdata_set = test_list{i};

            eval_type = eval_types{j};
            pred_type = pred_types{k};

            % load ground truth
            gtInfo = [save_dir train_sequence '_' eval_type '_' pred_type '/gt.txt'];
            gt = load(gtInfo);

            % load images list
            seq_dir =  [data_dir train_sequence '/img/img1/' ];
            im_list = dir([seq_dir, '*jpg']);
            im_list = strcat(seq_dir,{im_list.name});

            % get number of frames of sequence
            num_frames = length(im_list);

            % save visualizations here, exclude if not enough storage
            save_cluster_dir = [save_dir  train_sequence '_' eval_type '_' pred_type '/cluster_vis/'  '/full'];
            if ~exist(save_cluster_dir)
                mkdir(save_cluster_dir);
            end
            save_tracks_dir = [save_dir  train_sequence '_' eval_type '_' pred_type '/tracks_vis/'  '/full'];
            if ~exist(save_tracks_dir)
                mkdir(save_tracks_dir);
            end
            save_MOTA_dir = [save_dir train_sequence '_' eval_type '_' pred_type];

            % load boxes
            kboxes = load([save_dir train_sequence '_' eval_type '_' pred_type '/eval_input.txt']);
            idx=kboxes(:,2)>-1;

            % created boxes array from remaining detections
            boxes=[];
            boxes(:,1:2) = kboxes(:,3:4);
            boxes(:,3:4) = kboxes(:,3:4)+ kboxes(:,5:6);
            boxes(:,5) = kboxes(:,1)-1;
            boxes(:,6) = kboxes(:,2)+1;

            % load original detections
            cost_boxes = load([data_dir train_sequence '/det/det.txt']);
            cost_boxes = cost_boxes(idx,:)';
            oboxes = boxes(idx,:)';

            cost_boxes=-cost_boxes(7,1:end)';
            cluster_list = unique(boxes(:,6));
            clusters_all = convert_bbox_cluster(oboxes', -cost_boxes,minscore(i),minlscore(i));

            % evaluate
            fid = fopen([save_MOTA_dir '/record_all.txt'],'w');

            min_size_list = [ml(i)];
        %     inds=find(gt(:,8)==1 );
        %     gt=gt(inds,:);
        %     gtTracks={};
        %     tt=[gt(1,3) gt(1,4) gt(1,3)+ gt(1,5)  gt(1,4)+gt(1,6)  gt(1,1:2)];
        %     for le=2:size(gt,1)
        %         if(gt(le,2)== gt(le-1,2))
        %             tt=[tt; gt(le,3) gt(le,4) gt(le,5)+gt(le,3) gt(le,6)+gt(le,4)  gt(le,1:2)];
        %         else
        %             gtTracks{end+1}=tt;
        %             tt=[gt(le,3) gt(le,4) gt(le,5)+gt(le,3) gt(le,6)+gt(le,4)  gt(le,1:2)];
        %         end
        %     end
        %     gtTracks{end+1}=tt;
            bbx_scale=88;
            for m = 1:length(min_size_list)
                min_size = min_size_list(m);
                cluster_size_list = cellfun(@size, clusters_all,'UniformOutput',0);
                cluster_size_list = cell2mat(cluster_size_list');
                cluster_size_list = cluster_size_list(:,1);
                clusters = clusters_all(cluster_size_list(:) > min_size );

                tracks = convert_cluster_to_tracklet(clusters,min_size, 1);
                % submission
        %         fprintf(fid, '\n min_size %d; \n', min_size);
                tracks = fillingIn(tracks);
        %         evaluate_tracking_e(tracks,gtTracks,num_frames,fid);
                % visualization of the final result
        %         cluster_vis(tracks,im_list,save_cluster_dir);
            end
            fclose(fid);

            % convert to MOT evaluation format
            result = [];
            for t = 1:length(tracks)
                track = tracks{t};
                frame = track(:,5);
                x = track(:,1); y = track(:,2);
                w = (track(:,3) - track(:,1));
                h = (track(:,4) - track(:,2));
                pad =  repmat(-1, length(frame),4);
                id = repmat(t, length(frame),1);
                result{t} = [frame id x y w h pad];
            end
            result = cat(1, result{:});
            save_txt = [save_dir train_sequence '_' eval_type '_' pred_type '/' train_sequence '.txt'];
            dlmwrite(save_txt,result);
        end
    end
end
end

function [clusters] = convert_bbox_cluster(boxes,cost_boxes,minscore,minlscore)
boxes(:,5) = boxes(:,5) +1;
boxes(:,end+1) = cost_boxes;
cluster_list = unique(boxes(:,6));
clusters = cell(1,max(cluster_list));
labelWeights=[];
for f=1:max(boxes(:,5))
a=find(boxes(:,5)==f);
fboxes= boxes(a,:);
for i=1:length(a)
    b1 = (fboxes(:,4)>=fboxes(i,4));
    b2 = fboxes(:,6)>0;
    b=find(b1.*b2);
    heights = fboxes(b,4) - fboxes(b,2);
    avsize = mean(heights);
    stdheights=std(heights);
    if(fboxes(i,4) - fboxes(i,2) >2*avsize+0.5);
     %  boxes(a(i),6)=-1;
    %elseif(fboxes(i,4) - fboxes(i,2) <7);
     %    boxes(a(i),6)=-1;
    elseif cost_boxes(a(i))<minscore
   
         boxes(a(i),6)=-1;
    end 
end
end

max(cluster_list);
for c = 1:max(cluster_list) 
    tmp_list = find(boxes(:,6) == c);
    if(length(tmp_list)>0)
    labelweight = mean(cost_boxes(tmp_list));
    labelWeights(end+1)= labelweight;
    if(labelweight>minlscore)%4%-0.2
    clusters{c} = (boxes(tmp_list,1:6));
    end
    end
end

end
function box = interp(boxes)
[~,k] = max(boxes(:,end));
box = boxes(k,:);
end
function box = interp2(boxes)
b_tmp = sum(boxes(:,1:4).*repmat(boxes(:,6),[1,4]),1)/sum(boxes(:,6));
box = [b_tmp,boxes(1,5),mean(boxes(:,6))];
end


function evaluate_tracking_e(tracks,gtInfo,num_frame,fid)
tracks{1};
gtInfo{1};
%addpath /home/tang/Projects/multicut_tracking/tracking-multicut-git/utils
fprintf('evaluating tracking result... \n');

%evaluate tracking

stateInfo = convert_tracks_to_stateInfo(tracks,num_frame);
gtInfo = convert_tracks_to_stateInfo(gtInfo,num_frame);

list = find(sum(gtInfo.X,  1) == 0);
if ~isempty(list)
    gtInfo.X(:,list) = [];
    gtInfo.Y(:,list) = [];
    gtInfo.H(:,list) = [];
    gtInfo.W(:,list) = [];
end

[addInfo2d, id_list_all,frame_list] = printFinalEvaluation_e(stateInfo, gtInfo,fid);
end
%%
function stateInfo = convert_tracks_to_stateInfo(tracks,num_frame)
stateInfo = repmat(struct('Xi',[],'Yi',[],'W',[],'H',[] ), 1, 1);

num_track = length(tracks);
stateInfo.X = zeros(num_frame,num_track);
stateInfo.Y = zeros(num_frame,num_track);
stateInfo.Xi = zeros(num_frame,num_track);
stateInfo.Yi = zeros(num_frame,num_track);
stateInfo.W = zeros(num_frame,num_track);
stateInfo.H = zeros(num_frame,num_track);
stateInfo.S = zeros(num_frame,num_track);
for i =1 : length(tracks)

    tracklet = tracks{i};
    frame_id = uint64(tracklet(:,5));
    
    stateInfo.Xi(frame_id,i) = (tracklet(:,1) +  tracklet(:,3))/2;
    stateInfo.Yi(frame_id,i) = tracklet(:,4);
 stateInfo.X(frame_id,i) = (tracklet(:,1) +  tracklet(:,3))/2;
    stateInfo.Y(frame_id,i) = tracklet(:,4);  stateInfo.H(frame_id,i) = tracklet(:,4) - tracklet(:,2);
%     stateInfo.W(frame_idx,i) = tracklet(frame_id,3) - tracklet(frame_id,1);
    stateInfo.W(frame_id,i) = (tracklet(:,3) - tracklet(:,1));%(tracklet(:,4) - tracklet(:,2)).*88/200;
end
end
function [addInfo2d ,id_list_all,frame_list] = printFinalEvaluation_e(stateInfo,gtInfo,fid)


% zero metrics
[metrics2d metrics3d addInfo2d addInfo3d]= getMetricsForEmptySolution();
[metrics2d metricsInfo2d addInfo2d]=CLEAR_MOT_HUN(gtInfo,stateInfo);

% print to screen
printMetrics(metrics2d,metricsInfo2d,1);

% print to file
printMetrics_to_txt(metrics2d,metricsInfo2d,1,fid);

[ids,switched_tracklet_length,tracklet_length,id_list_all,frame_list] = printIDS_tracklet(addInfo2d.alltracked);
fprintf('swiched id in total: %d\n', ids );

% print to file
fprintf(fid,'\n swiched id in total:  %d\n', ids );

end
function tracklet = convert_cluster_to_tracklet(cluster_list,tracklet_window, smooth)

num_cluster = length(cluster_list);
count = 0;
tracklet=[];
for i = 1:num_cluster
    cluster = cluster_list{i};
    frame_list = unique(cluster(:,5));
    if length(frame_list) >= tracklet_window
        count = count+1;
        for f  = 1:length(frame_list)
            f_idx = frame_list(f);
            tmp_list = cluster(:,5) == f_idx;
            cluster(tmp_list,:);
            tmp_box = interp(cluster(tmp_list,:));          
           % tmp_box = interp2(cluster(tmp_list,:));
            tracklet{count}(f,:) = tmp_box;
        end
        %if(smooth)
        %tracklet{count} = smooth_tracklet3(tracklet{count}, smooth);
        %end
            
    end
end
end



function detections_s = smooth_tracklet(detections, is_smooth)
num_det = size(detections,1);
x_var = 1 : num_det;
ox = (detections(:,1) + detections(:,3))/2;
if is_smooth
    p = polyfit(x_var,ox',1);
    ox_new = polyval(p,x_var)';
else
    ox_new = ox;
end

y_var = 1 : num_det;
oy = (detections(:,2) + detections(:,4))/2;

if is_smooth
    p = polyfit(y_var,oy',1);
    oy_new = polyval(p,x_var)';
else
    oy_new = oy;
end

scale = detections(:,4) - detections(:,2);
p = polyfit(x_var,scale',3);
scale_new = polyval(p,x_var)';


w = scale_new.*80/200;
%w = detections(:,3) - detections(:,1);

x1 = ox_new-w/2;
x2 = ox_new+w/2;

y1 = oy_new - scale_new/2;
y2 = oy_new + scale_new/2;

detections_s = [x1 y1 x2 y2 detections(:,5:6)];
end

function detections_s = smooth_tracklet2(detections, is_smooth)
num_det = size(detections,1);
x_var = 1 : num_det;
ox = (detections(:,1) + detections(:,3))/2;

ox_new = smooth(detections(:,5),ox,'lowess');
%ox_new = ox;


y_var = 1 : num_det;
oy = (detections(:,2) + detections(:,4))/2;
oy_new = smooth(detections(:,5),oy,'lowess');

%oy_new = oy;

scale = detections(:,4) - detections(:,2);
p = polyfit(x_var,scale',2);
%scale_new = polyval(p,x_var)';

scale_new =smooth(detections(:,5),scale,'rlowess');
scale_new = scale;
if(is_smooth==0)
scale_new = scale;
oy_new = oy;
ox_new = ox;
end
w = (scale_new/scale) * (detections(:,3) - detections(:,1));
w=scale_new.*80/200;
%w = (detections(:,3) - detections(:,1));
x1 = ox_new-w/2;
x2 = ox_new+w/2;

y1 = oy_new - scale_new/2;
y2 = oy_new + scale_new/2;


detections_s = [x1 y1 x2 y2 detections(:,5:6)];
end
function detections = smooth_tracklet3(cluster, splineNum)
splineNum=4;
    detections = [];

    num_det = size(cluster,1);
    frame_list = cluster(:,5);
    frames = frame_list(1):frame_list(end);
    
    ox = (cluster(:,1) + cluster(:,3))/2;
    pp = splinefit(frame_list,ox,max(1,floor(num_det/splineNum)));
    ox_fit = ppval(pp,frames);
    
    
    oy = (cluster(:,2) + cluster(:,4))/2;
    pp = splinefit(frame_list,oy,max(1,floor(num_det/splineNum)));
    oy_fit = ppval(pp,frames);
    
    
    scale = cluster(:,4) - cluster(:,2);
    pp = splinefit(frame_list,scale,max(1,floor(num_det/splineNum)));
%     pp = splinefit(frame_list,scale,max(1,floor(num_det/splineNum)));
    scale_fit = ppval(pp,frames);
%     figure
%     plot(frame_list,scale,'.',frames, scale_fit);
%     
%     w = cluster(:,3) - cluster(:,1);
%     pp = splinefit(frame_list,w,max(1,floor(num_det/splineNum)));
%     w = ppval(pp,frames);

    w = scale_fit.*80/200;
%     w = max(w_min',w_fit')';
    
    x1 = ox_fit-w/2;
    x2 = ox_fit+w/2;

    y1 = oy_fit - scale_fit/2;
    y2 = oy_fit + scale_fit/2;

    detections = [x1' y1' x2' y2' frames'];
    detections(frame_list-frame_list(1)+1,6) = cluster(:,6);
   % detections(frame_list-frame_list(1)+1,7) = cluster(:,7);

end

function  cluster_vis(clusters,img_list,save_dir)
if ~exist(save_dir)
    mkdir(save_dir);
end

num_cluster = length(clusters);
if num_cluster >0
    track_colors = get_track_colors(num_cluster, 2);
    
    clusters_all = cat(1,clusters{:});
    frames_list = unique(clusters_all(:,5));
    for imgidx = frames_list(1):frames_list(end)
%     for imgidx = 37
        comp_cluster_vis_helper(img_list, clusters,  track_colors, imgidx,save_dir);
    end
end
end

function comp_cluster_vis_helper(img_list, clusters,  track_colors, aidx,save_dir)

drawArrow = @(x,y,varargin) quiver( x(1),y(1),x(2)-x(1),y(2)-y(1),0, varargin{:} );

text_x_offset = 10;
text_y_offset = 15;

figidx = 1;
% h = figure(figidx);
h = sfigure(figidx);
set(h,'Visible','off');
clf;

I = imread(img_list{aidx});
imshow(I, 'border', 'tight'); %axis equal; xlim([0 1000]);ylim([0 800]);
hold on;

for tidx = 1:length(clusters)
    fidx_list = clusters{tidx}(:,5);
    if any(fidx_list == aidx)
        bbox_idx_list = find(fidx_list == aidx);
        for k = 1: length(bbox_idx_list)
            bbox_idx = bbox_idx_list(k);%clusters{tidx}(bbox_idx_list(k));
            
            det = clusters{tidx}(bbox_idx,:);
            rect_x1 = max(det(1), 1);
            rect_y1 = max(det(2), 1);
            
            %             rect_x2 = min(det(3), size(I, 2));
            %             rect_y2 = min(det(4), size(I, 1));
            rect_x2 = det(3);
            rect_y2 = det(4);
            
            rect_width = rect_x2 - rect_x1;
            rect_height = rect_y2 - rect_y1;
            if rect_width>0 &&rect_height>0
            rectangle('Position', [rect_x1, rect_y1, rect_width, rect_height], ...
                'EdgeColor', track_colors(tidx, :), 'LineWidth', 5);
            text(rect_x1 + text_x_offset, rect_y1 - text_y_offset, ...
                num2str(tidx), 'FontSize', 15, 'Color', track_colors(tidx, :));hold on;
            end
            %% visualize track life-time
%             if bbox_idx>1
%                 tmp =[];
%                 for i =1:bbox_idx
%                     tmp_det = clusters{tidx}(i,:);
%                     tmp(i,:) = [(tmp_det(1)+tmp_det(3))/2, tmp_det(4)];
%                 end
%                 tmp(:,1) = smooth(tmp(:,1),'rlowess');
%                 tmp(:,2) = smooth(tmp(:,2),'rlowess');
%                 plot(tmp(:,1),tmp(:,2),'-','color',track_colors(tidx, :),'LineWidth', 2);
%             end
           %%
%             if (tidx == 6 && (aidx>=4&&aidx<=12) )|| (tidx == 5&& (aidx>=49&&aidx<=71) ) || (tidx == 2&& (aidx>=37&&aidx<=43) ) %|| (tidx == 19&& (aidx>=54&&aidx<=60) ) || (tidx == 40&& (aidx>=164&&aidx<=173) )
%                 x1 = [rect_x1+100, rect_x1];
%                 y1 = [rect_y1-100, rect_y1];
%                 drawArrow(x1,y1,'MaxHeadSize',10,'linewidth',5,'color', track_colors(tidx, :));
%             end
%              if (tidx == 6)|| (tidx == 1) || (tidx == 7 ) || (tidx == 3 ) || (tidx == 4 )%|| (tidx == 19&& (aidx>=54&&aidx<=60) ) || (tidx == 40&& (aidx>=164&&aidx<=173) )
%                 x1 = [(rect_x1+rect_width/2)+50, (rect_x1+rect_width/2)];
%                 y1 = [rect_y1-100, rect_y1];
%                 drawArrow(x1,y1,'MaxHeadSize',15,'linewidth',8,'color', track_colors(tidx, :));
%             end           
%              if (tidx == 2)|| (tidx == 5) %|| (tidx == 7)|| (tidx == 14)|| (tidx == 22)|| (tidx == 33)|| (tidx == 55)|| (tidx == 39)|| (tidx == 84)|| (tidx == 87) %|| (tidx == 19&& (aidx>=54&&aidx<=60) ) || (tidx == 40&& (aidx>=164&&aidx<=173) )
%                 x1 = [(rect_x1+rect_width/2)+50, (rect_x1+rect_width/2)];
%                 y1 = [rect_y1-100, rect_y1];
%                 drawArrow(x1,y1,'MaxHeadSize',15,'linewidth',8,'color', track_colors(tidx, :));
%             end  
        end
    end
end

frm = getframe( h );
vis_name = [save_dir '/imgidx' num2str(aidx,'%04.f') '.jpg'];
fprintf('saving %s\n', vis_name);
% set(gca, 'visible', 'off', 'position', [0, 0, 1, 1]);
% print(['-f' num2str(figidx)], '-djpeg', vis_name);
im = frm.cdata;
% im(:,end1: end+size(fr2,2),: )=frm2.cdata;
imwrite(im,vis_name);
end
function h = sfigure(h)
% SFIGURE  Create figure window (minus annoying focus-theft).
%
% Usage is identical to figure.
%
% Daniel Eaton, 2005
%
% See also figure

if nargin>=1 
	if ishandle(h)
		set(0, 'CurrentFigure', h);
	else
		h = figure(h);
	end
else
	h = figure;
end
end


