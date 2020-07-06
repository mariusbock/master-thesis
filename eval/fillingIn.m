function [tracklet_new] = fillingIn(tracklet)

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
        
%         labels = zeros(1, new_track_length);
%         labels(non_empty_entry) = 1;
        jump_list = find( diff(non_empty_entry) > 1);
        for j = 1 : length(jump_list)
            jumps = non_empty_entry(jump_list(j) + 1) -  non_empty_entry(jump_list(j)) - 1; % number of jumps
            s_idx = non_empty_entry(jump_list(j));
            e_idx = non_empty_entry(jump_list(j))+jumps+1;
            s_track = new_track(s_idx,:);
            e_track = new_track(e_idx,:);

            if jumps<30
                mmotions = [0 0 0 0 0 0 ];
                mmotione = [0 0 0 0 0 0 ];
                if s_idx>2 &&  new_track(s_idx-2,1)>0          
                    mmotions = (s_track - new_track(s_idx-2,:))/2;%motion before
                    mmotions = mmotions/norm(mmotions);
                end
                if e_idx<size(new_track,1)-2  &&  new_track(e_idx+2,1)>0  %motion after        
                    mmotione = -(e_track - new_track(e_idx+2,:))/2; 
                    mmotione = mmotione/norm(mmotione);
                end
                mmotion  =  (e_track-s_track)/jumps;
                mmotion = mmotion/norm(mmotion);


                %if sum(mmotion.*mmotions)>-0.9 && sum(mmotion.*mmotione)>-0.9
                    inter_tracks  = interpolate(s_track, e_track, jumps);   
                    new_track((s_idx+1) : (e_idx -1),: ) = inter_tracks;
                %else 
                %    breaklist{c}(1,end+1) = s_idx;
                %    breaklist{c}(2,end+1) = e_idx;
                %end
            else
                breaklist{c}(1,end+1) = s_idx;
                breaklist{c}(2,end+1) = e_idx;
%                 if(s_idx>=5)
%                new_track = new_track(1:s_idx,:);
%                tracklet{end+1} = new_track(e_idx:end,:);
%                disp('break');
%                num_tracklet=num_tracklet+1;
%                 break;
%                 else
%                     new_track = [];
%                     tracklet{end+1} = new_track(e_idx:end,:);
%                     disp('break');
%                     num_tracklet=num_tracklet+1;
%                     break;
%                     
%                 end
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

function   inter_track = interpolate(s_track, e_track, jumps)

x_1 = interp1([1, 1+jumps+1], [s_track(1),   e_track(1)], 2: jumps+1 );
y_1 = interp1([1, 1+jumps+1], [s_track(2),   e_track(2)], 2: jumps+1 );
x_2 = interp1([1, 1+jumps+1], [s_track(3),   e_track(3)], 2: jumps+1 );
y_2 = interp1([1, 1+jumps+1], [s_track(4),   e_track(4)], 2: jumps+1 );
t = interp1([1, 1+jumps+1], [s_track(5),   e_track(5)], 2: jumps+1 );
s = interp1([1, 1+jumps+1], [s_track(6),   e_track(6)], 2: jumps+1 );
inter_track = [x_1', y_1', x_2',y_2', t',s' ];
end


