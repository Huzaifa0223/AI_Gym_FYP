// Exercise database with filtering logic
// This matches the frontend exercise data structure

const exerciseDatabase = [
    // CHEST - Children (10-16)
    { name: 'Push-ups', muscleGroup: 'Chest', goal: ['gain', 'loss', 'maintenance'], level: 'easy', ageGroup: ['children'], equipment: 'None', description: 'Classic bodyweight chest exercise', reps: '12', sets: 3, restTime: 60, calories: 40, formTips: ['Keep back straight', 'Lower chest to ground', 'Elbows at 45° angle'] },
    { name: 'Incline Wall Push-ups', muscleGroup: 'Chest', goal: ['gain', 'loss', 'maintenance'], level: 'easy', ageGroup: ['children'], equipment: 'Wall', description: 'Safe chest exercise for youth', reps: '12', sets: 3, restTime: 60, calories: 35, formTips: ['Stand at wall', 'Push away from wall', 'Controlled pace'] },
    { name: 'Knee Push-ups', muscleGroup: 'Chest', goal: ['gain', 'loss', 'maintenance'], level: 'easy', ageGroup: ['children'], equipment: 'None', description: 'Modified push-ups for beginners', reps: '12', sets: 3, restTime: 60, calories: 35, formTips: ['Knees on ground', 'Keep back straight', 'Control movement'] },

    // CHEST - Adults easy (gain)
    { name: 'Flat Dumbbell Press', muscleGroup: 'Chest', goal: ['gain'], level: 'easy', ageGroup: ['adults'], equipment: 'Dumbbells, Bench', description: 'Basic chest mass builder', reps: '12', sets: 3, restTime: 90, calories: 60, formTips: ['Keep back flat', 'Lower until elbows at 90°', 'Press explosively'] },
    { name: 'Incline Dumbbell Press', muscleGroup: 'Chest', goal: ['gain'], level: 'easy', ageGroup: ['adults'], equipment: 'Dumbbells, Incline Bench', description: 'Upper chest focus', reps: '12', sets: 3, restTime: 90, calories: 65, formTips: ['30-45° incline', 'Full range of motion', 'Control weight'] },
    { name: 'Chest Press Machine', muscleGroup: 'Chest', goal: ['gain'], level: 'easy', ageGroup: ['adults'], equipment: 'Chest Press Machine', description: 'Guided chest movement', reps: '12', sets: 3, restTime: 90, calories: 65, formTips: ['Adjust seat height', 'Full extension', 'Control negative'] },

    // CHEST - Adults intermediate (gain)
    { name: 'Barbell Bench Press', muscleGroup: 'Chest', goal: ['gain'], level: 'intermediate', ageGroup: ['adults'], equipment: 'Barbell, Bench', description: 'Primary chest mass builder', reps: '15', sets: 3, restTime: 120, calories: 80, formTips: ['Grip slightly wider than shoulders', 'Touch chest', 'Drive through feet'] },
    { name: 'Cable Fly', muscleGroup: 'Chest', goal: ['gain'], level: 'intermediate', ageGroup: ['adults'], equipment: 'Cable Machine', description: 'Constant tension chest isolation', reps: '15', sets: 3, restTime: 60, calories: 65, formTips: ['Keep tension throughout', 'Squeeze at center', 'Slight forward lean'] },

    // CHEST - Adults hard (gain)
    { name: 'Barbell Bench Press (Heavy)', muscleGroup: 'Chest', goal: ['gain'], level: 'hard', ageGroup: ['adults'], equipment: 'Barbell, Bench', description: 'Maximum chest mass builder', reps: '15', sets: 4, restTime: 180, calories: 100, formTips: ['Heavy weight', 'Perfect form', 'Spotter recommended'] },
    { name: 'Weighted Dips', muscleGroup: 'Chest', goal: ['gain'], level: 'hard', ageGroup: ['adults'], equipment: 'Dip Bar, Weight Belt', description: 'Advanced chest and tricep builder', reps: '15', sets: 4, restTime: 120, calories: 90, formTips: ['Lean forward for chest', 'Full range', 'Add weight progressively'] },

    // BICEPS - Adults
    { name: 'Dumbbell Bicep Curls', muscleGroup: 'Biceps', goal: ['gain', 'loss', 'maintenance'], level: 'easy', ageGroup: ['adults', 'children'], equipment: 'Dumbbells', description: 'Classic bicep builder', reps: '12', sets: 3, restTime: 60, calories: 40, formTips: ['Keep elbows fixed', 'Full range of motion', 'Control the negative'] },
    { name: 'Hammer Curls', muscleGroup: 'Biceps', goal: ['gain', 'loss', 'maintenance'], level: 'easy', ageGroup: ['adults'], equipment: 'Dumbbells', description: 'Targets brachialis muscle', reps: '12', sets: 3, restTime: 60, calories: 40, formTips: ['Neutral grip', 'Keep elbows at sides', 'Controlled movement'] },
    { name: 'Barbell Curls', muscleGroup: 'Biceps', goal: ['gain'], level: 'intermediate', ageGroup: ['adults'], equipment: 'Barbell', description: 'Heavy bicep builder', reps: '15', sets: 3, restTime: 90, calories: 55, formTips: ['Shoulder width grip', 'No swinging', 'Squeeze at top'] },
    { name: 'Preacher Curls', muscleGroup: 'Biceps', goal: ['gain'], level: 'hard', ageGroup: ['adults'], equipment: 'Preacher Bench, Barbell', description: 'Strict bicep isolation', reps: '15', sets: 4, restTime: 90, calories: 60, formTips: ['Full stretch at bottom', 'Controlled pace', 'No momentum'] },

    // TRICEPS - Adults
    { name: 'Tricep Dips', muscleGroup: 'Triceps', goal: ['gain', 'loss', 'maintenance'], level: 'easy', ageGroup: ['adults', 'children'], equipment: 'Bench', description: 'Bodyweight tricep exercise', reps: '12', sets: 3, restTime: 60, calories: 45, formTips: ['Keep elbows close', 'Lower to 90°', 'Full extension'] },
    { name: 'Tricep Pushdowns', muscleGroup: 'Triceps', goal: ['gain', 'loss', 'maintenance'], level: 'easy', ageGroup: ['adults'], equipment: 'Cable Machine', description: 'Cable tricep isolation', reps: '12', sets: 3, restTime: 60, calories: 40, formTips: ['Keep elbows at sides', 'Full extension', 'Control return'] },
    { name: 'Skull Crushers', muscleGroup: 'Triceps', goal: ['gain'], level: 'intermediate', ageGroup: ['adults'], equipment: 'EZ Bar, Bench', description: 'Heavy tricep builder', reps: '15', sets: 3, restTime: 90, calories: 55, formTips: ['Lower to forehead', 'Keep elbows in', 'Control weight'] },
    { name: 'Close Grip Bench Press', muscleGroup: 'Triceps', goal: ['gain'], level: 'hard', ageGroup: ['adults'], equipment: 'Barbell, Bench', description: 'Compound tricep builder', reps: '15', sets: 4, restTime: 120, calories: 75, formTips: ['Hands shoulder width', 'Elbows close to body', 'Full lockout'] },

    // BACK - Adults
    { name: 'Lat Pulldowns', muscleGroup: 'Back', goal: ['gain', 'loss', 'maintenance'], level: 'easy', ageGroup: ['adults'], equipment: 'Cable Machine', description: 'Wide back builder', reps: '12', sets: 3, restTime: 90, calories: 55, formTips: ['Wide grip', 'Pull to chest', 'Control return'] },
    { name: 'Seated Rows', muscleGroup: 'Back', goal: ['gain', 'loss', 'maintenance'], level: 'easy', ageGroup: ['adults'], equipment: 'Cable Machine', description: 'Middle back thickness', reps: '12', sets: 3, restTime: 90, calories: 55, formTips: ['Keep back straight', 'Pull to abdomen', 'Squeeze shoulder blades'] },
    { name: 'Pull-ups', muscleGroup: 'Back', goal: ['gain'], level: 'intermediate', ageGroup: ['adults'], equipment: 'Pull-up Bar', description: 'King of back exercises', reps: '15', sets: 3, restTime: 120, calories: 70, formTips: ['Full hang', 'Chin over bar', 'Control descent'] },
    { name: 'Barbell Rows', muscleGroup: 'Back', goal: ['gain'], level: 'hard', ageGroup: ['adults'], equipment: 'Barbell', description: 'Heavy back builder', reps: '15', sets: 4, restTime: 120, calories: 80, formTips: ['45° forward lean', 'Pull to lower chest', 'Keep core tight'] },

    // LEGS - Adults
    { name: 'Bodyweight Squats', muscleGroup: 'Legs', goal: ['gain', 'loss', 'maintenance'], level: 'easy', ageGroup: ['adults', 'children'], equipment: 'None', description: 'Fundamental leg exercise', reps: '12', sets: 3, restTime: 60, calories: 50, formTips: ['Feet shoulder width', 'Knees track over toes', 'Full depth'] },
    { name: 'Lunges', muscleGroup: 'Legs', goal: ['gain', 'loss', 'maintenance'], level: 'easy', ageGroup: ['adults', 'children'], equipment: 'None', description: 'Single leg strength', reps: '12', sets: 3, restTime: 60, calories: 50, formTips: ['90° both knees', 'Keep torso upright', 'Push through heel'] },
    { name: 'Leg Press', muscleGroup: 'Legs', goal: ['gain'], level: 'intermediate', ageGroup: ['adults'], equipment: 'Leg Press Machine', description: 'Heavy quad builder', reps: '15', sets: 3, restTime: 120, calories: 75, formTips: ['Feet shoulder width', 'Full range of motion', 'Dont lock knees'] },
    { name: 'Barbell Squats', muscleGroup: 'Legs', goal: ['gain'], level: 'hard', ageGroup: ['adults'], equipment: 'Barbell, Squat Rack', description: 'King of leg exercises', reps: '15', sets: 4, restTime: 180, calories: 100, formTips: ['Bar on traps', 'Break at hips first', 'Full depth'] },

    // ABDOMINAL - Adults
    { name: 'Crunches', muscleGroup: 'Abdominal', goal: ['gain', 'loss', 'maintenance'], level: 'easy', ageGroup: ['adults', 'children'], equipment: 'None', description: 'Basic ab exercise', reps: '15', sets: 3, restTime: 45, calories: 30, formTips: ['Hands behind head', 'Curl up not pull', 'Exhale on crunch'] },
    { name: 'Plank', muscleGroup: 'Abdominal', goal: ['gain', 'loss', 'maintenance'], level: 'easy', ageGroup: ['adults', 'children'], equipment: 'None', description: 'Core stability', reps: '30 seconds', sets: 3, restTime: 60, calories: 25, formTips: ['Straight line from head to heels', 'Engage core', 'Dont drop hips'], isTimeBased: true },
    { name: 'Hanging Leg Raises', muscleGroup: 'Abdominal', goal: ['gain'], level: 'intermediate', ageGroup: ['adults'], equipment: 'Pull-up Bar', description: 'Lower ab focus', reps: '15', sets: 3, restTime: 60, calories: 45, formTips: ['Controlled movement', 'Dont swing', 'Full range'] },
    { name: 'Ab Wheel Rollout', muscleGroup: 'Abdominal', goal: ['gain'], level: 'hard', ageGroup: ['adults'], equipment: 'Ab Wheel', description: 'Advanced core exercise', reps: '15', sets: 4, restTime: 90, calories: 55, formTips: ['Keep core tight', 'Full extension', 'Control return'] },

    // SHOULDERS - Adults
    { name: 'Dumbbell Shoulder Press', muscleGroup: 'Shoulders', goal: ['gain', 'loss', 'maintenance'], level: 'easy', ageGroup: ['adults'], equipment: 'Dumbbells', description: 'Basic shoulder builder', reps: '12', sets: 3, restTime: 90, calories: 55, formTips: ['Press straight up', 'Dont lock elbows', 'Control weight'] },
    { name: 'Lateral Raises', muscleGroup: 'Shoulders', goal: ['gain', 'loss', 'maintenance'], level: 'easy', ageGroup: ['adults'], equipment: 'Dumbbells', description: 'Side delt isolation', reps: '12', sets: 3, restTime: 60, calories: 40, formTips: ['Slight elbow bend', 'Raise to shoulder height', 'Control descent'] },
    { name: 'Military Press', muscleGroup: 'Shoulders', goal: ['gain'], level: 'intermediate', ageGroup: ['adults'], equipment: 'Barbell', description: 'Compound shoulder builder', reps: '15', sets: 3, restTime: 120, calories: 70, formTips: ['Strict form', 'Full lockout', 'Brace core'] },
    { name: 'Arnold Press', muscleGroup: 'Shoulders', goal: ['gain'], level: 'hard', ageGroup: ['adults'], equipment: 'Dumbbells', description: 'Advanced shoulder exercise', reps: '15', sets: 4, restTime: 90, calories: 65, formTips: ['Rotate as you press', 'Full range', 'Control movement'] }
];

// Helper function to get age group
function getAgeGroup(age) {
    if (age >= 10 && age <= 16) return 'children';
    if (age >= 17 && age <= 45) return 'adults';
    return 'older';
}

// Helper function to get level based on experience months
function getLevel(experienceMonths) {
    if (experienceMonths < 6) return 'easy';
    if (experienceMonths < 18) return 'intermediate';
    return 'hard';
}

// @desc    Get filtered exercises
// @route   GET /api/exercises
// @access  Private
exports.getExercises = async (req, res) => {
    try {
        const { muscleGroup, goal, age, experienceMonths } = req.query;

        // Use user data if not provided in query
        const userAge = age || req.user.age;
        const userExp = experienceMonths || req.user.experienceMonths;
        const userGoal = goal || req.user.goal;

        const ageGroup = getAgeGroup(parseInt(userAge));
        const level = getLevel(parseInt(userExp));

        let filtered = exerciseDatabase;

        // Filter by muscle group
        if (muscleGroup && muscleGroup !== 'Rest') {
            // Handle combined muscle groups like "Biceps/Triceps"
            const groups = muscleGroup.split('/').map(g => g.trim());
            filtered = filtered.filter(e => groups.includes(e.muscleGroup));
        }

        // Filter by goal
        if (userGoal) {
            filtered = filtered.filter(e => e.goal.includes(userGoal));
        }

        // Filter by level
        filtered = filtered.filter(e => e.level === level);

        // Filter by age group
        filtered = filtered.filter(e => e.ageGroup.includes(ageGroup));

        res.json({
            exercises: filtered,
            count: filtered.length,
            filters: {
                muscleGroup,
                goal: userGoal,
                level,
                ageGroup
            }
        });
    } catch (error) {
        console.error('Get exercises error:', error);
        res.status(500).json({ message: 'Server error' });
    }
};

// @desc    Get all exercises (unfiltered)
// @route   GET /api/exercises/all
// @access  Private
exports.getAllExercises = async (req, res) => {
    try {
        res.json({
            exercises: exerciseDatabase,
            count: exerciseDatabase.length
        });
    } catch (error) {
        console.error('Get all exercises error:', error);
        res.status(500).json({ message: 'Server error' });
    }
};
