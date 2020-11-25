function postprocessing_script
    % POSTPROCESSING SCRIPT
    % This script can be used to convert predictions made by the GCN to
    % trajectories. This is a modified version of the one provided by 
    % Professor Margret Keuper. 
    % PARAMETERS:
    %   log_dir   --   directory where log files are located 
    %                  (i.e. eval_input.txt and heuristic_input.txt)
    %   img_dir   --   directory where image files of sequences are located
    %   type      --   type of prediction ('full' or 'removed')
    %   runs      --   number of runs in log directory
    %   testing   --   boolean whether testing files are to be handled
    %   visualize --   boolean whether to visualize results
    %   heuristic --   boolean whether to use labels proposed by external 
    %                  heuristic  
    %   minscore  --   minimum detector confidence score (all detections)
    %   minlscore --   minimum average detector confidence score in 
    %                  trajectory (cluster)
    %   min_size  --   shortest trajectory of pedestrian that is accepted 
    %                  (cluster size)
    %   gap_size  --   maximum frame gap size
    %   is_smooth --   boolean whether to apply smoothing on sequences
    
    % PARAMETERS
    log_dir = '../logs/20201113/000000';
    img_dir = '../data/MOT/MOT17';
    type = 'full'; 
    runs = 1; 
    testing = false; 
    visualize = true; 
    heuristic = true; 

     % minimum detector confidence score (all detections)
%      minscore = [+0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 ... SDP
%                  -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 ... DPM
%                  +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 ... FRCNN
%                  ];

     % minimum detector confidence score (all detections)
     minscore = [-1 -1 -1 -1 -1 -1 -1 ... SDP
                 -1 -1 -1 -1 -1 -1 -1 ... DPM
                 -1 -1 -1 -1 -1 -1 -1 ... FRCNN
                 ];

     % minimum average detector confidence score in trajectory (cluster)
%      minlscore = [0.5 0.5 0.5 0.5 0.5 0.5 0.5 ... SDP
%                   0.2 0.2 0.2 0.2 0.2 0.2 0.2 ... DPM
%                   0.5 0.5 0.5 0.5 0.5 0.5 0.5 ... FRCNN
%                   ];
     minlscore = [-1 -1 -1 -1 -1 -1 -1 ... SDP
                  -1 -1 -1 -1 -1 -1 -1 ... DPM
                  -1 -1 -1 -1 -1 -1 -1 ... FRCNN
                  ];

    
     % shortest trajectory of pedestrian that is accepted (cluster size)
%      min_size = [5 5 5 5 5 5 5 ... SDP
%                  5 5 5 5 5 5 5 ... DPM
%                  5 5 5 5 5 5 5 ... FRCNN
%                  ];
    min_size = [3 3 3 3 3 3 3 ... SDP
                2 2 2 2 2 2 2 ... DPM
                3 3 3 3 3 3 3 ... FRCNN
                ];
            
    % maximum frame gap size
    gap_size = 30;
             
    % apply smoothing (boolean)
    is_smooth = [0 0 0 0 0 0 0 ... SDP
                 0 0 0 0 0 0 0 ... DPM
                 0 0 0 0 0 0 0 ... FRCNN
                 ];
%     is_smooth = [1 1 1 1 1 1 1 ... SDP
%                  1 1 1 1 1 1 1 ... DPM
%                  1 1 1 1 1 1 1 ... FRCNN
%                  ];


    if testing
        img_dir = [img_dir '/test'];
        sequence_list = {
             'MOT17-01-SDP' 'MOT17-03-SDP' 'MOT17-06-SDP' 'MOT17-07-SDP' ... SDP
             'MOT17-08-SDP' 'MOT17-12-SDP' 'MOT17-14-SDP' ... SDP
             'MOT17-01-DPM' 'MOT17-03-DPM' 'MOT17-06-DPM' 'MOT17-07-DPM' ... DPM
             'MOT17-08-DPM' 'MOT17-12-DPM' 'MOT17-14-DPM' ... DPM
             'MOT17-01-FRCNN' 'MOT17-03-FRCNN' 'MOT17-06-FRCNN' 'MOT17-07-FRCNN' ... FRCNN
             'MOT17-08-FRCNN' 'MOT17-12-FRCNN' 'MOT17-14-FRCNN' ... FRCNN
             };
    else
        img_dir = [img_dir '/train'];
        sequence_list = {
             'MOT17-02-SDP' 'MOT17-04-SDP' 'MOT17-05-SDP' 'MOT17-09-SDP' ... SDP
             'MOT17-10-SDP' 'MOT17-11-SDP' 'MOT17-13-SDP' ... SDP
             'MOT17-02-DPM' 'MOT17-04-DPM' 'MOT17-05-DPM' 'MOT17-09-DPM' ... DPM
             'MOT17-10-DPM' 'MOT17-11-DPM' 'MOT17-13-DPM' ... DPM
             'MOT17-02-FRCNN' 'MOT17-04-FRCNN' 'MOT17-05-FRCNN' 'MOT17-09-FRCNN' ... FRCNN
             'MOT17-10-FRCNN' 'MOT17-11-FRCNN' 'MOT17-13-FRCNN' ... FRCNN
             };
    end
   
    for i = 1:length(sequence_list)
        for run = 1:(runs)
            sequence = sequence_list{i};
            % load images list
            seq_dir =  [img_dir '/' sequence '/img/img1/' ];
            im_list = dir([seq_dir, '*jpg']);
            im_list = strcat(seq_dir,{im_list.name});
            
            % load boxes
            kboxes = load([log_dir '/run_' int2str(run) '/' type '/' sequence '/eval_input.txt']);
            if heuristic
                labels = load([log_dir '/run_' int2str(run) '/' type '/' sequence '/heuristic_output.txt']);
            end
            
            % created boxes array from remaining detections
            boxes=[];
            boxes(:,1:2) = kboxes(:,3:4);
            boxes(:,3:4) = kboxes(:,3:4) + kboxes(:,5:6);
            boxes(:,5) = kboxes(:,1) - 1;
            if heuristic
                boxes(:,6) = labels + 1;
            else
                boxes(:,6) = kboxes(:,2) + 1;
            end

            % filter out detections with label -1
            idx=kboxes(:,2)>-1;
            % load original detections
            cost_boxes = load([img_dir '/' sequence '/det/det.txt']);
            cost_boxes = cost_boxes(idx,:)';
            oboxes = boxes(idx,:)';

            cost_boxes=-cost_boxes(7,1:end)';
            % returns list of lists (contains all detections of a cluster)
            clusters_all = convert_bbox_cluster(oboxes', -cost_boxes, minscore(i), minlscore(i));

            % keep only those clusters of minimum size 
            cluster_size_list = cellfun(@size, clusters_all,'UniformOutput',0);
            cluster_size_list = cell2mat(cluster_size_list');
            cluster_size_list = cluster_size_list(:,1);
            clusters = clusters_all(cluster_size_list(:) > min_size(i));

            % convert clusters to tracks
            tracks = convert_cluster_to_tracklet(clusters, min_size(i), is_smooth(i));
            tracks = fillingIn(tracks, gap_size);
                
            % visualization of the final result
            if visualize
                save_cluster_dir = [log_dir '/run_' int2str(run) '/' type '/' sequence '/cluster_vis'];
                cluster_vis(tracks, im_list, save_cluster_dir);
            end

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
            save_txt = [log_dir '/run_' int2str(run) '/' type '/' sequence '/' sequence '.txt'];
            dlmwrite(save_txt,result);
        end
    end
end

function [clusters] = convert_bbox_cluster(boxes, cost_boxes, minscore, minlscore)
    % increase label by 1
    boxes(:,5) = boxes(:,5) +1;
    % append detector confidence
    boxes(:,end+1) = cost_boxes;
    % obtain cluster labels
    cluster_list = unique(boxes(:,6));
    % empty list of clusters
    clusters = cell(1,max(cluster_list));
    labelWeights=[];
    % go frame by frame
    for f=1:max(boxes(:,5))
        % all boxes that are in said frame
        a=find(boxes(:,5)==f);
        fboxes= boxes(a,:);
        % for each box
        for i=1:length(a)
            % all bounding boxes y2 larger (boolean array)
            b1 = (fboxes(:,4)>=fboxes(i,4));
            % all bounding boxes with no negative frame
            b2 = fboxes(:,6)>0;
            % both combined (index returned)
            b=find(b1.*b2);
            % get height
            heights = fboxes(b,4) - fboxes(b,2);
            % average and std height
            avsize = mean(heights);
            stdheights=std(heights);
            if(fboxes(i,4) - fboxes(i,2) >2*avsize+0.5);
             %  boxes(a(i),6)=-1;
            %elseif(fboxes(i,4) - fboxes(i,2) <7);
             %    boxes(a(i),6)=-1;
            % if detector is below minscore confident exclude detection
            elseif cost_boxes(a(i))<minscore
                boxes(a(i),6)=-1;
            end 
        end
    end

    max(cluster_list);
    % iterate through all cluster labels
    for c = 1:max(cluster_list) 
        % find all boxes in cluster
        tmp_list = find(boxes(:,6) == c);
        % if size of cluster is above 0
        if(length(tmp_list)>0)
            % average detector confidence
            labelweight = mean(cost_boxes(tmp_list));
            % append average confidence to list
            labelWeights(end+1)= labelweight;
            % if above minlscore then append to cluster list
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

%%

function tracklet = convert_cluster_to_tracklet(cluster_list, tracklet_window, smooth)
    num_cluster = length(cluster_list);
    count = 0;
    tracklet=[];
    for i = 1:num_cluster
        cluster = cluster_list{i};
        frame_list = unique(cluster(:,5));
        if length(frame_list) >= tracklet_window
            count = count+1;
            for f = 1:length(frame_list)
                f_idx = frame_list(f);
                tmp_list = cluster(:,5) == f_idx;
                cluster(tmp_list,:);
                tmp_box = interp(cluster(tmp_list,:));          
                %tmp_box = interp2(cluster(tmp_list,:));
                tracklet{count}(f,:) = tmp_box;
            end
            if(smooth)
                tracklet{count} = smooth_tracklet3(tracklet{count});
            end
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

function detections = smooth_tracklet3(cluster)
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

function cluster_vis(clusters,img_list,log_dir)
    if ~exist(log_dir)
        mkdir(log_dir);
    end
    num_cluster = length(clusters);
    if num_cluster >0
        track_colors = get_track_colors(num_cluster, 2);
        clusters_all = cat(1,clusters{:});
        frames_list = unique(clusters_all(:,5));
        for imgidx = frames_list(1):frames_list(end)
            comp_cluster_vis_helper(img_list, clusters,  track_colors, imgidx,log_dir);
        end
    end
end

function comp_cluster_vis_helper(img_list, clusters, track_colors, aidx, log_dir)
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
    vis_name = [log_dir '/imgidx' num2str(aidx,'%04.f') '.jpg'];
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

function track_colors = get_track_colors(num_tracks, rnd_seed)
    track_colors2 = hsv(3*num_tracks);
    rand('twister', rnd_seed);
    rp = randperm(size(track_colors2, 1));
    for idx = 1:num_tracks
        track_colors(idx, :) = track_colors2(rp(idx), :);
    end
end

function [tracklet_new] = fillingIn(tracklet, gap_size)
    num_tracklet = length(tracklet);
    breaklist = {};
    for c = 1:num_tracklet
        breaklist{c}=[];
        track = tracklet{c};
        [~,d] = size(track);
        frame_list = track(:,5);
        if length(frame_list) < frame_list(end)-frame_list(1)+1
            new_track_length = frame_list(end)-frame_list(1)+1;
            new_track = zeros(new_track_length,d);
            new_track(frame_list-frame_list(1)+1,:) = track;
            non_empty_entry = find(new_track(:,5) ~= 0);
            jump_list = find( diff(non_empty_entry) > 1);
            for j = 1 : length(jump_list)
                jumps = non_empty_entry(jump_list(j) + 1) -  non_empty_entry(jump_list(j)) - 1; % number of jumps
                s_idx = non_empty_entry(jump_list(j));
                e_idx = non_empty_entry(jump_list(j))+jumps+1;
                s_track = new_track(s_idx,:);
                e_track = new_track(e_idx,:);
                
                if jumps<gap_size
                    inter_tracks  = interpolate(s_track, e_track, jumps);   
                    new_track((s_idx+1) : (e_idx -1),: ) = inter_tracks;
                else
                    breaklist{c}(1,end+1) = s_idx;
                    breaklist{c}(2,end+1) = e_idx;
                end
            end
            if(size(new_track,1)>0)
                new_track(:,5) = round(new_track(:,5));
            end
            tracklet_new{c} = new_track;
        else
            tracklet_new{c} = track;
        end
    end
    for c = 1:num_tracklet
        if(length(breaklist{c})>0)  
            for b = 2:(length(breaklist{c}))
                if(breaklist{c}(1,b)-breaklist{c}(2,b-1)>=12)
                    tracklet_new{end+1}=tracklet_new{c}(breaklist{c}(2,b-1):breaklist{c}(1,b),:);
                end
            end
            if(abs(breaklist{c}(2,end)-size(tracklet_new{c},1))>=12)
                    tracklet_new{end+1}=tracklet_new{c}(breaklist{c}(2,end):end,:);
            end
            if(breaklist{c}(1,1)>=12)
                tracklet_new{c}=tracklet_new{c}(1:breaklist{c}(1,1),:);
            else
                tracklet_new{c}=[];
            end
        end
    end
    num_tracklet = length(tracklet_new);
    c=1;
    while c <=length(tracklet_new)
        if size(tracklet_new{c},1)==0;
            for j=c:length(tracklet_new)-1
               tracklet_new{j}= tracklet_new{j+1};
            end
             tracklet_new =  tracklet_new(1:end-1);
        else
            c=c+1;
        end
    end
end

function inter_track = interpolate(s_track, e_track, jumps)
    x_1 = interp1([1, 1+jumps+1], [s_track(1),   e_track(1)], 2: jumps+1 );
    y_1 = interp1([1, 1+jumps+1], [s_track(2),   e_track(2)], 2: jumps+1 );
    x_2 = interp1([1, 1+jumps+1], [s_track(3),   e_track(3)], 2: jumps+1 );
    y_2 = interp1([1, 1+jumps+1], [s_track(4),   e_track(4)], 2: jumps+1 );
    t = interp1([1, 1+jumps+1], [s_track(5),   e_track(5)], 2: jumps+1 );
    s = interp1([1, 1+jumps+1], [s_track(6),   e_track(6)], 2: jumps+1 );
    inter_track = [x_1', y_1', x_2', y_2', t',s' ];
end

function pp = splinefit(varargin)
    %SPLINEFIT Fit a spline to noisy data.
    %   PP = SPLINEFIT(X,Y,BREAKS) fits a piecewise cubic spline with breaks
    %   (knots) BREAKS to the noisy data (X,Y). X is a vector and Y is a vector
    %   or an ND array. If Y is an ND array, then X(j) and Y(:,...,:,j) are
    %   matched. Use PPVAL to evaluate PP.
    %
    %   PP = SPLINEFIT(X,Y,P) where P is a positive integer interpolates the
    %   breaks linearly from the sorted locations of X. P is the number of
    %   spline pieces and P+1 is the number of breaks.
    %
    %   OPTIONAL INPUT
    %   Argument places 4 to 8 are reserved for optional input.
    %   These optional arguments can be given in any order:
    %
    %   PP = SPLINEFIT(...,'p') applies periodic boundary conditions to
    %   the spline. The period length is MAX(BREAKS)-MIN(BREAKS).
    %
    %   PP = SPLINEFIT(...,'r') uses robust fitting to reduce the influence
    %   from outlying data points. Three iterations of weighted least squares
    %   are performed. Weights are computed from previous residuals.
    %
    %   PP = SPLINEFIT(...,BETA), where 0 < BETA < 1, sets the robust fitting
    %   parameter BETA and activates robust fitting ('r' can be omitted).
    %   Default is BETA = 1/2. BETA close to 0 gives all data equal weighting.
    %   Increase BETA to reduce the influence from outlying data. BETA close
    %   to 1 may cause instability or rank deficiency.
    %
    %   PP = SPLINEFIT(...,N) sets the spline order to N. Default is a cubic
    %   spline with order N = 4. A spline with P pieces has P+N-1 degrees of
    %   freedom. With periodic boundary conditions the degrees of freedom are
    %   reduced to P.
    %
    %   PP = SPLINEFIT(...,CON) applies linear constraints to the spline.
    %   CON is a structure with fields 'xc', 'yc' and 'cc':
    %       'xc', x-locations (vector)
    %       'yc', y-values (vector or ND array)
    %       'cc', coefficients (matrix).
    %
    %   Constraints are linear combinations of derivatives of order 0 to N-2
    %   according to
    %
    %     cc(1,j)*y(x) + cc(2,j)*y'(x) + ... = yc(:,...,:,j),  x = xc(j).
    %
    %   The maximum number of rows for 'cc' is N-1. If omitted or empty 'cc'
    %   defaults to a single row of ones. Default for 'yc' is a zero array.
    %
    %   EXAMPLES
    %
    %       % Noisy data
    %       x = linspace(0,2*pi,100);
    %       y = sin(x) + 0.1*randn(size(x));
    %       % Breaks
    %       breaks = [0:5,2*pi];
    %
    %       % Fit a spline of order 5
    %       pp = splinefit(x,y,breaks,5);
    %
    %       % Fit a spline of order 3 with periodic boundary conditions
    %       pp = splinefit(x,y,breaks,3,'p');
    %
    %       % Constraints: y(0) = 0, y'(0) = 1 and y(3) + y"(3) = 0
    %       xc = [0 0 3];
    %       yc = [0 1 0];
    %       cc = [1 0 1; 0 1 0; 0 0 1];
    %       con = struct('xc',xc,'yc',yc,'cc',cc);
    %
    %       % Fit a cubic spline with 8 pieces and constraints
    %       pp = splinefit(x,y,8,con);
    %
    %       % Fit a spline of order 6 with constraints and periodicity
    %       pp = splinefit(x,y,breaks,con,6,'p');
    %
    %   See also SPLINE, PPVAL, PPDIFF, PPINT
    %   Author: Jonas Lundgren <splinefit@gmail.com> 2010
    %   2009-05-06  Original SPLINEFIT.
    %   2010-06-23  New version of SPLINEFIT based on B-splines.
    %   2010-09-01  Robust fitting scheme added.
    %   2010-09-01  Support for data containing NaNs.
    %   2011-07-01  Robust fitting parameter added.
    % Check number of arguments
    error(nargchk(3,7,nargin));
    % Check arguments
    [x,y,dim,breaks,n,periodic,beta,constr] = arguments(varargin{:});
    % Evaluate B-splines
    base = splinebase(breaks,n);
    pieces = base.pieces;
    A = ppval(base,x);
    % Bin data
    [junk,ibin] = histc(x,[-inf,breaks(2:end-1),inf]); %#ok
    % Sparse system matrix
    mx = numel(x);
    ii = [ibin; ones(n-1,mx)];
    ii = cumsum(ii,1);
    jj = repmat(1:mx,n,1);
    if periodic
        ii = mod(ii-1,pieces) + 1;
        A = sparse(ii,jj,A,pieces,mx);
    else
        A = sparse(ii,jj,A,pieces+n-1,mx);
    end
    % Don't use the sparse solver for small problems
    if pieces < 20*n/log(1.7*n)
        A = full(A);
    end
    % Solve
    if isempty(constr)
        % Solve Min norm(u*A-y)
        u = lsqsolve(A,y,beta);
    else
        % Evaluate constraints
        B = evalcon(base,constr,periodic);
        % Solve constraints
        [Z,u0] = solvecon(B,constr);
        % Solve Min norm(u*A-y), subject to u*B = yc
        y = y - u0*A;
        A = Z*A;
        v = lsqsolve(A,y,beta);
        u = u0 + v*Z;
    end
    % Periodic expansion of solution
    if periodic
        jj = mod(0:pieces+n-2,pieces) + 1;
        u = u(:,jj);
    end
    % Compute polynomial coefficients
    ii = [repmat(1:pieces,1,n); ones(n-1,n*pieces)];
    ii = cumsum(ii,1);
    jj = repmat(1:n*pieces,n,1);
    C = sparse(ii,jj,base.coefs,pieces+n-1,n*pieces);
    coefs = u*C;
    coefs = reshape(coefs,[],n);
    % Make piecewise polynomial
    pp = mkpp(breaks,coefs,dim);
end
%--------------------------------------------------------------------------
function [x,y,dim,breaks,n,periodic,beta,constr] = arguments(varargin)
    %ARGUMENTS Lengthy input checking
    %   x           Noisy data x-locations (1 x mx)
    %   y           Noisy data y-values (prod(dim) x mx)
    %   dim         Leading dimensions of y
    %   breaks      Breaks (1 x (pieces+1))
    %   n           Spline order
    %   periodic    True if periodic boundary conditions
    %   beta        Robust fitting parameter, no robust fitting if beta = 0
    %   constr      Constraint structure
    %   constr.xc   x-locations (1 x nx)
    %   constr.yc   y-values (prod(dim) x nx)
    %   constr.cc   Coefficients (?? x nx)
    % Reshape x-data
    x = varargin{1};
    mx = numel(x);
    x = reshape(x,1,mx);
    % Remove trailing singleton dimensions from y
    y = varargin{2};
    dim = size(y);
    while numel(dim) > 1 && dim(end) == 1
        dim(end) = [];
    end
    my = dim(end);
    % Leading dimensions of y
    if numel(dim) > 1
        dim(end) = [];
    else
        dim = 1;
    end
    % Reshape y-data
    pdim = prod(dim);
    y = reshape(y,pdim,my);
    % Check data size
    if mx ~= my
        mess = 'Last dimension of array y must equal length of vector x.';
        error('arguments:datasize',mess)
    end
    % Treat NaNs in x-data
    inan = find(isnan(x));
    if ~isempty(inan)
        x(inan) = [];
        y(:,inan) = [];
        mess = 'All data points with NaN as x-location will be ignored.';
        warning('arguments:nanx',mess)
    end
    % Treat NaNs in y-data
    inan = find(any(isnan(y),1));
    if ~isempty(inan)
        x(inan) = [];
        y(:,inan) = [];
        mess = 'All data points with NaN in their y-value will be ignored.';
        warning('arguments:nany',mess)
    end
    % Check number of data points
    mx = numel(x);
    if mx == 0
        error('arguments:nodata','There must be at least one data point.')
    end
    % Sort data
    if any(diff(x) < 0)
        [x,isort] = sort(x);
        y = y(:,isort);
    end
    % Breaks
    if isscalar(varargin{3})
        % Number of pieces
        p = varargin{3};
        if ~isreal(p) || ~isfinite(p) || p < 1 || fix(p) < p
            mess = 'Argument #3 must be a vector or a positive integer.';
            error('arguments:breaks1',mess)
        end
        if x(1) < x(end)
            % Interpolate breaks linearly from x-data
            dx = diff(x);
            ibreaks = linspace(1,mx,p+1);
            [junk,ibin] = histc(ibreaks,[0,2:mx-1,mx+1]); %#ok
            breaks = x(ibin) + dx(ibin).*(ibreaks-ibin);
        else
            breaks = x(1) + linspace(0,1,p+1);
        end
    else
        % Vector of breaks
        breaks = reshape(varargin{3},1,[]);
        if isempty(breaks) || min(breaks) == max(breaks)
            mess = 'At least two unique breaks are required.';
            error('arguments:breaks2',mess);
        end
    end
    % Unique breaks
    if any(diff(breaks) <= 0)
        breaks = unique(breaks);
    end
    % Optional input defaults
    n = 4;                      % Cubic splines
    periodic = false;           % No periodic boundaries
    robust = false;             % No robust fitting scheme
    beta = 0.5;                 % Robust fitting parameter
    constr = [];                % No constraints
    % Loop over optional arguments
    for k = 4:nargin
        a = varargin{k};
        if ischar(a) && isscalar(a) && lower(a) == 'p'
            % Periodic conditions
            periodic = true;
        elseif ischar(a) && isscalar(a) && lower(a) == 'r'
            % Robust fitting scheme
            robust = true;
        elseif isreal(a) && isscalar(a) && isfinite(a) && a > 0 && a < 1
            % Robust fitting parameter
            beta = a;
            robust = true;
        elseif isreal(a) && isscalar(a) && isfinite(a) && a > 0 && fix(a) == a
            % Spline order
            n = a;
        elseif isstruct(a) && isscalar(a)
            % Constraint structure
            constr = a;
        else
            error('arguments:nonsense','Failed to interpret argument #%d.',k)
        end
    end
    % No robust fitting
    if ~robust
        beta = 0;
    end
    % Check exterior data
    h = diff(breaks);
    xlim1 = breaks(1) - 0.01*h(1);
    xlim2 = breaks(end) + 0.01*h(end);
    if x(1) < xlim1 || x(end) > xlim2
        if periodic
            % Move data inside domain
            P = breaks(end) - breaks(1);
            x = mod(x-breaks(1),P) + breaks(1);
            % Sort
            [x,isort] = sort(x);
            y = y(:,isort);
        else
            mess = 'Some data points are outside the spline domain.';
            warning('arguments:exteriordata',mess)
        end
    end
    % Return
    if isempty(constr)
        return
    end
    % Unpack constraints
    xc = [];
    yc = [];
    cc = [];
    names = fieldnames(constr);
    for k = 1:numel(names)
        switch names{k}
            case {'xc'}
                xc = constr.xc;
            case {'yc'}
                yc = constr.yc;
            case {'cc'}
                cc = constr.cc;
            otherwise
                mess = 'Unknown field ''%s'' in constraint structure.';
                warning('arguments:unknownfield',mess,names{k})
        end
    end
    % Check xc
    if isempty(xc)
        mess = 'Constraints contains no x-locations.';
        error('arguments:emptyxc',mess)
    else
        nx = numel(xc);
        xc = reshape(xc,1,nx);
    end
    % Check yc
    if isempty(yc)
        % Zero array
        yc = zeros(pdim,nx);
    elseif numel(yc) == 1
        % Constant array
        yc = zeros(pdim,nx) + yc;
    elseif numel(yc) ~= pdim*nx
        % Malformed array
        error('arguments:ycsize','Cannot reshape yc to size %dx%d.',pdim,nx)
    else
        % Reshape array
        yc = reshape(yc,pdim,nx);
    end
    % Check cc
    if isempty(cc)
        cc = ones(size(xc));
    elseif numel(size(cc)) ~= 2
        error('arguments:ccsize1','Constraint coefficients cc must be 2D.')
    elseif size(cc,2) ~= nx
        mess = 'Last dimension of cc must equal length of xc.';
        error('arguments:ccsize2',mess)
    end
    % Check high order derivatives
    if size(cc,1) >= n
        if any(any(cc(n:end,:)))
            mess = 'Constraints involve derivatives of order %d or larger.';
            error('arguments:difforder',mess,n-1)
        end
        cc = cc(1:n-1,:);
    end
    % Check exterior constraints
    if min(xc) < xlim1 || max(xc) > xlim2
        if periodic
            % Move constraints inside domain
            P = breaks(end) - breaks(1);
            xc = mod(xc-breaks(1),P) + breaks(1);
        else
            mess = 'Some constraints are outside the spline domain.';
            warning('arguments:exteriorconstr',mess)
        end
    end
    % Pack constraints
    constr = struct('xc',xc,'yc',yc,'cc',cc);
end
%--------------------------------------------------------------------------
function pp = splinebase(breaks,n)
    %SPLINEBASE Generate B-spline base PP of order N for breaks BREAKS
    breaks = breaks(:);     % Breaks
    breaks0 = breaks';      % Initial breaks
    h = diff(breaks);       % Spacing
    pieces = numel(h);      % Number of pieces
    deg = n - 1;            % Polynomial degree
    % Extend breaks periodically
    if deg > 0
        if deg <= pieces
            hcopy = h;
        else
            hcopy = repmat(h,ceil(deg/pieces),1);
        end
        % to the left
        hl = hcopy(end:-1:end-deg+1);
        bl = breaks(1) - cumsum(hl);
        % and to the right
        hr = hcopy(1:deg);
        br = breaks(end) + cumsum(hr);
        % Add breaks
        breaks = [bl(deg:-1:1); breaks; br];
        h = diff(breaks);
        pieces = numel(h);
    end
    % Initiate polynomial coefficients
    coefs = zeros(n*pieces,n);
    coefs(1:n:end,1) = 1;
    % Expand h
    ii = [1:pieces; ones(deg,pieces)];
    ii = cumsum(ii,1);
    ii = min(ii,pieces);
    H = h(ii(:));
    % Recursive generation of B-splines
    for k = 2:n
        % Antiderivatives of splines
        for j = 1:k-1
            coefs(:,j) = coefs(:,j).*H/(k-j);
        end
        Q = sum(coefs,2);
        Q = reshape(Q,n,pieces);
        Q = cumsum(Q,1);
        c0 = [zeros(1,pieces); Q(1:deg,:)];
        coefs(:,k) = c0(:);
        % Normalize antiderivatives by max value
        fmax = repmat(Q(n,:),n,1);
        fmax = fmax(:);
        for j = 1:k
            coefs(:,j) = coefs(:,j)./fmax;
        end
        % Diff of adjacent antiderivatives
        coefs(1:end-deg,1:k) = coefs(1:end-deg,1:k) - coefs(n:end,1:k);
        coefs(1:n:end,k) = 0;
    end
    % Scale coefficients
    scale = ones(size(H));
    for k = 1:n-1
        scale = scale./H;
        coefs(:,n-k) = scale.*coefs(:,n-k);
    end
    % Reduce number of pieces
    pieces = pieces - 2*deg;
    % Sort coefficients by interval number
    ii = [n*(1:pieces); deg*ones(deg,pieces)];
    ii = cumsum(ii,1);
    coefs = coefs(ii(:),:);
    % Make piecewise polynomial
    pp = mkpp(breaks0,coefs,n);
end
%--------------------------------------------------------------------------
function B = evalcon(base,constr,periodic)
    %EVALCON Evaluate linear constraints
    % Unpack structures
    breaks = base.breaks;
    pieces = base.pieces;
    n = base.order;
    xc = constr.xc;
    cc = constr.cc;
    % Bin data
    [junk,ibin] = histc(xc,[-inf,breaks(2:end-1),inf]); %#ok
    % Evaluate constraints
    nx = numel(xc);
    B0 = zeros(n,nx);
    for k = 1:size(cc,1)
        if any(cc(k,:))
            B0 = B0 + repmat(cc(k,:),n,1).*ppval(base,xc);
        end
        % Differentiate base
        coefs = base.coefs(:,1:n-k);
        for j = 1:n-k-1
            coefs(:,j) = (n-k-j+1)*coefs(:,j);
        end
        base.coefs = coefs;
        base.order = n-k;
    end
    % Sparse output
    ii = [ibin; ones(n-1,nx)];
    ii = cumsum(ii,1);
    jj = repmat(1:nx,n,1);
    if periodic
        ii = mod(ii-1,pieces) + 1;
        B = sparse(ii,jj,B0,pieces,nx);
    else
        B = sparse(ii,jj,B0,pieces+n-1,nx);
    end
end
%--------------------------------------------------------------------------
function [Z,u0] = solvecon(B,constr)
    %SOLVECON Find a particular solution u0 and null space Z (Z*B = 0)
    %         for constraint equation u*B = yc.
    yc = constr.yc;
    tol = 1000*eps;
    % Remove blank rows
    ii = any(B,2);
    B2 = full(B(ii,:));
    % Null space of B2
    if isempty(B2)
        Z2 = [];
    else
        % QR decomposition with column permutation
        [Q,R,dummy] = qr(B2); %#ok
        R = abs(R);
        jj = all(R < R(1)*tol, 2);
        Z2 = Q(:,jj)';
    end
    % Sizes
    [m,ncon] = size(B);
    m2 = size(B2,1);
    nz = size(Z2,1);
    % Sparse null space of B
    Z = sparse(nz+1:nz+m-m2,find(~ii),1,nz+m-m2,m);
    Z(1:nz,ii) = Z2;
    % Warning rank deficient
    if nz + ncon > m2
        mess = 'Rank deficient constraints, rank = %d.';
        warning('solvecon:deficient',mess,m2-nz);
    end
    % Particular solution
    u0 = zeros(size(yc,1),m);
    if any(yc(:))
        % Non-homogeneous case
        u0(:,ii) = yc/B2;
        % Check solution
        if norm(u0*B - yc,'fro') > norm(yc,'fro')*tol
            mess = 'Inconsistent constraints. No solution within tolerance.';
            error('solvecon:inconsistent',mess)
        end
    end
end
%--------------------------------------------------------------------------
function u = lsqsolve(A,y,beta)
    %LSQSOLVE Solve Min norm(u*A-y)
    % Avoid sparse-complex limitations
    if issparse(A) && ~isreal(y)
        A = full(A);
    end
    % Solution
    u = y/A;
    % Robust fitting
    if beta > 0
        [m,n] = size(y);
        alpha = 0.5*beta/(1-beta)/m;
        for k = 1:3
            % Residual
            r = u*A - y;
            rr = r.*conj(r);
            rrmean = sum(rr,2)/n;
            rrmean(~rrmean) = 1;
            rrhat = (alpha./rrmean)'*rr;
            % Weights
            w = exp(-rrhat);
            spw = spdiags(w',0,n,n);
            % Solve weighted problem
            u = (y*spw)/(A*spw);
        end
    end
end